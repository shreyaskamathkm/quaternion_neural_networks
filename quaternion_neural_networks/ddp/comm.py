# File   : comm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import collections
import queue
import threading

__all__ = ["FutureResult", "SyncPrimary", "WorkerPipe"]


class FutureResult:
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_PrimaryRegistry = collections.namedtuple("_PrimaryRegistry", ["result"])
_WorkerPipeBase = collections.namedtuple("_WorkerPipeBase", ["identifier", "queue", "result"])


class WorkerPipe(_WorkerPipeBase):
    """Pipe for primary-worker communication."""

    def run_worker(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncPrimary:
    """An abstract `SyncPrimary` object.

    - During the replication, as the data parallel will trigger an callback of each module, all worker devices should
    call `register(id)` and obtain an `WorkerPipe` to communicate with the primary.
    - During the forward pass, primary device invokes `run_primary`, all messages from worker devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the primary device should gather the information and determine to message passed
    back to each worker devices.
    """

    def __init__(self, primary_callback):
        """

        Args:
            primary_callback: a callback to be invoked after having collected messages from worker devices.
        """
        self._primary_callback = primary_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def register_worker(self, identifier):
        """
        Register an worker device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `WorkerPipe` object which can be used to communicate with the primary device.

        """
        if self._activated:
            assert self._queue.empty(), "Queue is not clean before next initialization."
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _PrimaryRegistry(future)
        return WorkerPipe(identifier, self._queue, future)

    def run_primary(self, primary_msg):
        """
        Main entry for the primary device in each forward pass.
        The messages were first collected from each devices (including the primary device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the primary device).

        Args:
            primary_msg: the message that the primary want to send to itself. This will be placed as the first
            message when calling `primary_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the primary device.

        """
        self._activated = True

        intermediates = [(0, primary_msg)]
        for _i in range(self.nr_workers):
            intermediates.append(self._queue.get())

        results = self._primary_callback(intermediates)
        assert results[0][0] == 0, "The first result should belongs to the primary."

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for _i in range(self.nr_workers):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_workers(self):
        return len(self._registry)
