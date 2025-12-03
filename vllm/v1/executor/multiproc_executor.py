# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import signal
import sys
import time
import queue
import threading
import traceback
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Optional, Union
from threading import Thread
import cloudpickle
import psutil
import zmq

from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.executor.multiproc_worker_utils import (
    _add_prefix, set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_mp_context,
                        get_open_port, get_open_zmq_ipc_path, zmq_socket_ctx)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import (AsyncModelRunnerOutput, ModelRunnerOutput)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000


class MultiprocExecutor(Executor):

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.shutdown_event = threading.Event()
        self.io_thread_pool : Optional[ThreadPoolExecutor] = None
        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen.
        def sigusr1_handler(signum, frame):
            logger.fatal(
                "MulitprocExecutor got fatal signal from worker processes, "
                "shutting down. See stack trace above for root cause issue.")
            # Propagate error up to parent process.
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert self.world_size == tensor_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        self.workers: list[WorkerProcHandle] = []
        for rank in range(self.world_size):
            worker = WorkerProc.make_worker_process(self.vllm_config, rank,
                                                    rank,
                                                    distributed_init_method,
                                                    scheduler_output_handle)
            self.workers.append(worker)

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc
        self.rpc_broadcast_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()

        if self.max_concurrent_batches > 1:
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")
        
        self.output_rank = self._get_output_rank()
        

    def execute_model(self, 
                      scheduler_output:SchedulerOutput,
                      non_block: bool = False,
        ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        (output, )  = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            unique_reply_rank=self.output_rank,
            non_block=non_block
        )
        logger.info(f"+++++++++++++++\noutput type {output}")
        return output


    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        
        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, unique_reply_rank))

            # responses = [None] * self.world_size
            responses = []
            workers = (self.workers[unique_reply_rank], 
                       ) if unique_reply_rank is not None else self.workers
            
            def get_response(w: WorkerProcHandle, 
                            dequeue_timeout: Optional[float] = None,
                            cancel_event:Optional[threading.Event] = None):
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout, cancel=cancel_event)
                
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        "Worker failed with error %s, please check the"
                        " stack trace above for the root cause", result)
                return result
            

            # for w in self.workers:
            for w in workers:
                dequeue_timeout = None if deadline is None else (
                    deadline - time.monotonic())
                if self.io_thread_pool is not None:
                    result = self.io_thread_pool.submit(
                        get_response, w, dequeue_timeout, self.shutdown_event
                    )
                    logger.info(f"-------------\nresult type {type(result)}")
                    if not non_block:
                        result = result.result()
                elif not non_block:
                    result = get_response(w, dequeue_timeout, self.shutdown_event)
                else:
                    raise RuntimeError("non_block can only be used when"
                                       " max_concurrent_batches > 1")
                # status, result = w.worker_response_mq.dequeue(
                #     timeout=dequeue_timeout)

                # if status != WorkerProc.ResponseStatus.SUCCESS:
                #     raise RuntimeError(
                #         "Worker failed with error %s, please check the"
                #         " stack trace above for the root cause", result)
                logger.info(f"-------------\nresult type {result}")
                
                # responses[w.rank] = result
                responses.append(result)

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except Exception as e:
            # Re-raise any other exceptions
            raise e

    def _ensure_worker_termination(self):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [w.proc for w in self.workers if w.proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

        self._cleanup_sockets()

    def _cleanup_sockets(self):
        for w in self.workers:
            # Remove the zmq ipc socket file
            socket_path = w.ready_path.replace("ipc://", "")
            if os and os.path.exists(socket_path):
                os.remove(socket_path)

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            for w in self.workers:
                w.worker_response_mq = None
            self._ensure_worker_termination()

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @cached_property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            logger.info(f"$$$$$$$$$$$$$$$$$$$$$$$")
            return 2
        return self.parallel_config.pipeline_parallel_size
    
    def _get_output_rank(self) -> int:
        # In multiproc executor, each worker has its own rank.
        # We assume the output rank is 0 for simplicity.
        return self.world_size - self.parallel_config.pipeline_parallel_size

@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    ready_path: str
    worker_response_mq: MessageQueue  # The worker process writes to this MQ


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        ready_path: str,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": rank == 0,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        pid = os.getpid()
        _add_prefix(sys.stdout, f"VllmWorker rank={rank}", pid)
        _add_prefix(sys.stderr, f"VllmWorker rank={rank}", pid)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)

        schduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = schduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy"
            )
            self.async_output_copy_thread.start()
        

        worker_response_mq_handle = self.worker_response_mq.export_handle()

        # Send Readiness signal to EngineCore process.
        # Set linger here because we want to ensure the message has
        # been sent before the context is closed.
        with zmq_socket_ctx(ready_path, zmq.constants.PUSH,
                            linger=10000) as ready_socket:
            payload = pickle.dumps(worker_response_mq_handle,
                                   protocol=pickle.HIGHEST_PROTOCOL)
            ready_socket.send_string(WorkerProc.READY_STR)
            ready_socket.send(payload)

        self.worker.init_device()
        self.worker.load_model()

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
    ) -> WorkerProcHandle:
        context = get_mp_context()

        # ZMQ path for worker to send ready message and shm_broadcast handle
        # back to core process.
        ready_path = get_open_zmq_ipc_path()

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_path": ready_path,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               name=f"VllmWorker-{rank}",
                               daemon=True)

        with zmq_socket_ctx(ready_path, zmq.constants.PULL) as ready_socket:
            proc.start()

            # Wait for startup
            worker_response_mq_handle = WorkerProc.wait_for_startup(
                proc, ready_socket)

        worker_response_mq = MessageQueue.create_from_handle(
            worker_response_mq_handle, 0)

        return WorkerProcHandle(proc, rank, ready_path, worker_response_mq)

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        shutdown_event = threading.Event()
        worker = None
        try:
            worker = WorkerProc(*args, **kwargs)

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()

            worker.worker_busy_loop(cancel=shutdown_event)

        except SystemExit:
            logger.debug("Worker interrupted.")

        except Exception:
            # worker_busy_loop sends exceptions to Executor
            # for shutdown, but if there is an error in startup or an
            # error with IPC itself, we need to alert the parent.
            psutil.Process().parent().send_signal(signal.SIGUSR1)
            raise

        finally:
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()
                worker = None

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_socket: zmq.Socket,
    ) -> Optional[Handle]:
        """Wait until the Worker is ready."""

        # Wait for Worker to send READY.
        while ready_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
            logger.debug("Waiting for WorkerProc to startup.")

            if not proc.is_alive():
                raise RuntimeError("WorkerProc failed to start.")

        message = ready_socket.recv_string()
        assert message == WorkerProc.READY_STR
        handle_frame = ready_socket.recv(copy=False)
        handle = pickle.loads(handle_frame.buffer)
        return handle

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, output):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """   
        if isinstance(output, AsyncModelRunnerOutput):
            logger.info(f"9999999999\n getting output from AsyncModelRunnerOutput, {output}")
            output = output.get_output()
            
        
        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))

        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if(response_mq := self.worker_response_mq) is not None:
            logger.info(f"-------------\nenqueuing output of type {(result)}")
            response_mq.enqueue(result)
    
    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output) 
    
    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            output = self.async_output_queue.get()
            logger.info(f"-------------\nasync output of type {output}")
            self.enqueue_output(output)
    
    def worker_busy_loop(self, cancel: Optional[threading.Event] = None):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            # method, args, kwargs = self.rpc_broadcast_mq.dequeue()
            method, args, kwargs, output_rank  = self.rpc_broadcast_mq.dequeue(
                cancel=cancel)

            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception: %s", exc_info=e)
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                # self.worker_response_mq.enqueue(
                #     (WorkerProc.ResponseStatus.FAILURE, str(e)))
                if output_rank is None or output_rank == self.rank:
                    self.handle_output(e)
                continue

            # self.worker_response_mq.enqueue(
            #     (WorkerProc.ResponseStatus.SUCCESS, output))
            if output_rank is None or output_rank == self.rank:
                self.handle_output(output)
