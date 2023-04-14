import io
import os
import sys
import traceback
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple
from zipfile import ZipFile

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from alphafed import get_result_dir, logger
from alphafed.data_channel.shared_file_data_channel import \
    SharedFileDataChannel
from alphafed.scheduler import Scheduler

from .contractor import (CheckinEvent, CheckinResponseEvent, CloseRoundEvent,
                         ReadyForAggregationEvent, SimpleFedAvgContractor,
                         StartRoundEvent)


class SimpleFedAvgScheduler(Scheduler, metaclass=ABCMeta):
    """A simple FedAvg implementation as an example of customized scheduler."""

    _INIT = 'init'
    _GETHORING = 'gethering'
    _READY = 'ready'
    _IN_A_ROUND = 'in_a_round'
    _UPDATING = 'updating'
    _CALCULATING = 'calculating'
    _WAIT_FOR_AGGR = 'wait_4_aggr'
    _AGGREGATING = 'aggregating'
    _PERSISTING = 'persisting'
    _CLOSING_ROUND = 'closing_round'
    _FINISHING = 'finishing'

    def __init__(self, clients: int, rounds: int):
        """Init.

        Args:
            clients:
                The number of calculators.
            rounds:
                The number of training rounds.
        """
        super().__init__()
        self.state = self._INIT

        self.clients = clients
        self.rounds = rounds

        self._participants = []

    @abstractmethod
    def build_model(self) -> Module:
        """Return a model object which will be used for training."""

    @property
    def model(self) -> Module:
        """Get the model object which is used for training."""
        if not hasattr(self, '_model'):
            self._model = self.build_model()
        return self._model

    @abstractmethod
    def build_optimizer(self, model: Module) -> Optimizer:
        """Return a optimizer object which will be used for training.

        Args:
            model:
                The model object which is used for training.
        """

    @property
    def optimizer(self) -> Optimizer:
        """Get the optimizer object which is used for training."""
        if not hasattr(self, '_optimizer'):
            self._optimizer = self.build_optimizer(model=self.model)
        return self._optimizer

    @abstractmethod
    def build_train_dataloader(self) -> DataLoader:
        """Define the training dataloader.

        You can transform the dataset, do some preprocess to the dataset.

        Return:
            training dataloader
        """

    @property
    def train_loader(self) -> DataLoader:
        """Get the training dataloader object."""
        if not hasattr(self, '_train_loader'):
            self._train_loader = self.build_train_dataloader()
        return self._train_loader

    @abstractmethod
    def build_test_dataloader(self) -> DataLoader:
        """Define the testing dataloader.

        You can transform the dataset, do some preprocess to the dataset. If you do not
        want to do testing after training, simply make it return None.

        Args:
            dataset:
                training dataset
        Return:
            testing dataloader
        """

    @property
    def test_loader(self) -> DataLoader:
        """Get the testing dataloader object."""
        if not hasattr(self, '_test_loader'):
            self._test_loader = self.build_test_dataloader()
        return self._test_loader

    @abstractmethod
    def state_dict(self) -> Dict[str, Tensor]:
        """Get the params that need to train and update.

        Only the params returned by this function will be updated and saved during aggregation.

        Return:
            List[Tensor], The list of model params.
        """

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Tensor]):
        """Load the params that trained and updated.

        Only the params returned by state_dict() should be loaded by this function.
        """

    @abstractmethod
    def train_an_epoch(self):
        """Define the training steps in an epoch."""

    @abstractmethod
    def test(self):
        """Define the testing steps.

        If you do not want to do testing after training, simply make it pass.
        """

    def _setup_context(self, id: str, task_id: str, is_initiator: bool = False):
        assert id, 'must specify a unique id for every participant'
        assert task_id, 'must specify a task_id for every participant'

        self.id = id
        self.task_id = task_id
        self._result_dir = get_result_dir(self.task_id)
        self._log_dir = os.path.join(self._result_dir, 'tb_logs')
        self.tb_writer = SummaryWriter(log_dir=self._log_dir)

        self.is_initiator = is_initiator

        self.contractor = SimpleFedAvgContractor(task_id=task_id)
        self.data_channel = SharedFileDataChannel(self.contractor)
        self.model
        self.optimizer
        self.round = 0

    def _run(self, id: str, task_id: str, is_initiator: bool = False, recover: bool = False):
        self._setup_context(id=id, task_id=task_id, is_initiator=is_initiator)
        self.push_log(message='Local context is ready.')
        try:
            if self.is_initiator and recover:
                self._recover_progress()
            else:
                self._clean_progress()
            self._launch_process()
        except Exception:
            # 将错误信息推送到 Playground 前端界面，有助于了解错误原因并修正
            err_stack = '\n'.join(traceback.format_exception(*sys.exc_info()))
            self.push_log(err_stack)

    def _recover_progress(self):
        """Try to recover and continue from last running."""
        # 如果上一次执行计算任务因为某些偶发原因失败了。在排除故障原因后，希望能够从失败的地方
        # 恢复计算进度继续计算，而不是重新开始，可以在这里提供恢复进度的处理逻辑。
        pass

    def _clean_progress(self):
        """Clean existing progress data."""
        # 如果曾经执行过计算任务，在计算环境中留下了一些过往的痕迹。现在想要从头开始重新运行计算
        # 任务，但是残留的数据可能会干扰当前这一次运行，可以在这里提供清理环境的处理逻辑。
        pass

    def _launch_process(self):
        self.push_log(f'Node {self.id} is up.')

        self._switch_status(self._GETHORING)
        self._check_in()

        self._switch_status(self._READY)
        self.round = 1

        for _ in range(self.rounds):
            self._switch_status(self._IN_A_ROUND)
            self._run_a_round()
            self._switch_status(self._READY)
            self.round += 1

        if self.is_initiator:
            self.push_log(f'Obtained the final results of task {self.task_id}')
            self._switch_status(self._FINISHING)
            self.test()
            self._close_task()

    def _check_in(self):
        """Check in task and get ready.

        As an initiator (and default the aggregator), records each participants
        and launches training process.
        As a participant, checkins and gets ready for training.
        """
        if self.is_initiator:
            self.push_log('Waiting for participants taking part in ...')
            self._wait_for_gathering()
        else:
            is_checked_in = False
            # the aggregator may be in special state so can not response
            # correctly nor in time, then retry periodically
            self.push_log('Checking in the task ...')
            while not is_checked_in:
                is_checked_in = self._check_in_task()
            self.push_log(f'Node {self.id} have taken part in the task.')

    def _wait_for_gathering(self):
        """Wait for participants gethering."""
        logger.debug('_wait_for_gathering ...')
        for _event in self.contractor.contract_events():
            if isinstance(_event, CheckinEvent):
                if _event.peer_id not in self._participants:
                    self._participants.append(_event.peer_id)
                    self.push_log(f'Welcome a new participant ID: {_event.peer_id}.')
                    self.push_log(f'There are {len(self._participants)} participants now.')
                self.contractor.respond_check_in(round=self.round,
                                                 aggregator=self.id,
                                                 nonce=_event.nonce,
                                                 requester_id=_event.peer_id)
                if len(self._participants) == self.clients:
                    break
        self.push_log('All participants gethered.')

    def _check_in_task(self) -> bool:
        """Try to check in the task."""
        nonce = self.contractor.checkin(peer_id=self.id)
        logger.debug('_wait_for_check_in_response ...')
        for _event in self.contractor.contract_events(timeout=30):
            if isinstance(_event, CheckinResponseEvent):
                if _event.nonce != nonce:
                    continue
                self.round = _event.round
                self._aggregator = _event.aggregator
                return True
        return False

    def _run_a_round(self):
        """Perform a round of FedAvg calculation.

        As an aggregator, selects a part of participants as actual calculators
        in the round, distributes latest parameters to them, collects update and
        makes aggregation.
        As a participant, if is selected as a calculator, calculates and uploads
        parameter update.
        """
        if self.is_initiator:
            self._run_as_aggregator()
        else:
            self._run_as_data_owner()

    def _run_as_aggregator(self):
        self._start_round()
        self._distribute_model()
        self._process_aggregation()
        self._close_round()

    def _start_round(self):
        """Prepare and start calculation of a round."""
        self.push_log(f'Begin the training of round {self.round}.')
        self.contractor.start_round(round=self.round,
                                    calculators=self._participants,
                                    aggregator=self.id)
        self.push_log(f'Calculation of round {self.round} is started.')

    def _distribute_model(self):
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        self.push_log('Distributing parameters ...')
        accept_list = self.data_channel.batch_send_stream(source=self.id,
                                                          target=self._participants,
                                                          data_stream=buffer.getvalue())
        self.push_log(f'Successfully distributed parameters to: {accept_list}')
        if len(self._participants) != len(accept_list):
            reject_list = [_target for _target in self._participants
                           if _target not in accept_list]
            self.push_log(f'Failed to distribute parameters to: {reject_list}')
            raise RuntimeError('Failed to distribute parameters to some participants.')
        self.push_log('Distributed parameters to all participants.')

    def _process_aggregation(self):
        """Process aggregation depending on specific algorithm."""
        self._switch_status(self._WAIT_FOR_AGGR)
        self.contractor.notify_ready_for_aggregation(round=self.round)
        self.push_log('Now waiting for executing calculation ...')
        accum_result, result_count = self._wait_for_calculation()
        if result_count < self.clients:
            self.push_log('Task failed because some calculation results lost.')
            raise RuntimeError('Task failed because some calculation results lost.')
        self.push_log(f'Received {result_count} copies of calculation results.')

        self._switch_status(self._AGGREGATING)
        self.push_log('Begin to aggregate and update parameters.')
        for _key in accum_result.keys():
            if accum_result[_key].dtype in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
            ):
                logger.warn(f'average a int value may lose precision: {_key=}')
                accum_result[_key].div_(result_count, rounding_mode='trunc')
            else:
                accum_result[_key].div_(result_count)
        self.load_state_dict(accum_result)
        self.push_log('Obtained a new version of parameters.')

    def _wait_for_calculation(self) -> Tuple[Dict[str, torch.Tensor], int]:
        """Wait for every calculator finish its task or timeout."""
        result_count = 0
        accum_result = self.state_dict()
        for _param in accum_result.values():
            if isinstance(_param, torch.Tensor):
                _param.zero_()

        self.push_log('Waiting for training results ...')
        training_results = self.data_channel.batch_receive_stream(
            receiver=self.id,
            source_list=self._participants
        )
        for _source, _result in training_results.items():
            buffer = io.BytesIO(_result)
            _new_state_dict = torch.load(buffer)
            for _key in accum_result.keys():
                accum_result[_key].add_(_new_state_dict[_key])
            result_count += 1
            self.push_log(f'Received calculation results from ID: {_source}')
        return accum_result, result_count

    def _close_round(self):
        """Close current round when finished."""
        self._switch_status(self._CLOSING_ROUND)
        self.contractor.close_round(round=self.round)
        self.push_log(f'The training of Round {self.round} complete.')

    def _run_as_data_owner(self):
        self._wait_for_starting_round()
        self._switch_status(self._UPDATING)
        self._wait_for_updating_model()

        self._switch_status(self._CALCULATING)
        self.push_log('Begin to run calculation ...')
        self.train_an_epoch()
        self.push_log('Local calculation complete.')

        self._wait_for_uploading_model()
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        self.push_log('Pushing local update to the aggregator ...')
        self.data_channel.send_stream(source=self.id,
                                      target=self._aggregator,
                                      data_stream=buffer.getvalue())
        self.push_log('Successfully pushed local update to the aggregator.')
        self._switch_status(self._CLOSING_ROUND)
        self._wait_for_closing_round()

        self.push_log(f'ID: {self.id} finished training task of round {self.round}.')

    def _wait_for_starting_round(self):
        """Wait for starting a new round of training."""
        self.push_log(f'Waiting for training of round {self.round} begin ...')
        for _event in self.contractor.contract_events():
            if isinstance(_event, StartRoundEvent):
                self.push_log(f'Training of round {self.round} begins.')
                return

    def _wait_for_updating_model(self):
        """Wait for receiving latest parameters from aggregator."""
        self.push_log('Waiting for receiving latest parameters from the aggregator ...')
        _, parameters = self.data_channel.receive_stream(receiver=self.id,
                                                         source=self._aggregator)
        buffer = io.BytesIO(parameters)
        new_state_dict = torch.load(buffer)
        self.load_state_dict(new_state_dict)
        self.push_log('Successfully received latest parameters.')
        return

    def _wait_for_uploading_model(self):
        """Wait for uploading trained parameters to aggregator."""
        self.push_log('Waiting for aggregation begin ...')
        for _event in self.contractor.contract_events():
            if isinstance(_event, ReadyForAggregationEvent):
                return

    def _wait_for_closing_round(self):
        """Wait for closing current round of training."""
        self.push_log(f'Waiting for closing signal of training round {self.round} ...')
        for _event in self.contractor.contract_events():
            if isinstance(_event, CloseRoundEvent):
                return

    def _close_task(self, is_succ: bool = True):
        """Close the FedAvg calculation.

        As an aggregator, broadcasts the finish task event to all participants,
        uploads the final parameters and tells L1 task manager the task is complete.
        As a participant, do nothing.
        """
        self.push_log(f'Closing task {self.task_id} ...')
        if self.is_initiator:
            self._switch_status(self._FINISHING)
            report_file_path, model_file_path = self._prepare_task_output()
            self.contractor.upload_metric_report(receivers=self.contractor.EVERYONE,
                                                 report_file=report_file_path)
            self.contractor.upload_model(receivers=self.contractor.EVERYONE,
                                         model_file=model_file_path)
            self.contractor.notify_task_completion(result=True)
        self.push_log(f'Task {self.task_id} closed. Byebye!')

    def _prepare_task_output(self) -> Tuple[str, str]:
        """Generate final output files of the task.

        Return:
            Local paths of the report file and model file.
        """
        self.push_log('Uploading task achievement and closing task ...')

        os.makedirs(self._result_dir, exist_ok=True)

        report_file = os.path.join(self._result_dir, "report.zip")
        with ZipFile(report_file, 'w') as report_zip:
            for path, _, filenames in os.walk(self._log_dir):
                rel_dir = os.path.relpath(path=path, start=self._result_dir)
                rel_dir = rel_dir.lstrip('.')  # ./file => file
                for _file in filenames:
                    rel_path = os.path.join(rel_dir, _file)
                    report_zip.write(os.path.join(path, _file), rel_path)
        report_file_path = os.path.abspath(report_file)

        model_file = os.path.join(self._result_dir, "model.pt")
        with open(model_file, 'wb') as f:
            torch.save(self.state_dict(), f)
        model_file_path = os.path.abspath(model_file)

        self.push_log('Task achievement files are ready.')
        return report_file_path, model_file_path
