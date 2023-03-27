import io
import os
from abc import ABCMeta, abstractmethod
from tempfile import TemporaryFile
from zipfile import ZipFile

import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from alphafed.data_channel import SharedFileDataChannel
from alphafed.fs import get_result_dir
from alphafed.scheduler import Scheduler

from .contractor import (CheckInEvent, CheckInResponseEvent, CloseEvent,
                         SimpleFedAvgContractor, StartEvent)


class SimpleFedAvgScheduler(Scheduler, metaclass=ABCMeta):

    def __init__(self, rounds: int) -> None:
        super().__init__()
        # 自定义一些初始化参数，此处只定义了 rounds 一个参数，用于设置训练的轮数
        self.rounds = rounds

    def _run(self, id: str, task_id: str, is_initiator: bool = False, recover: bool = False):
        """运行调度器的入口。

        实际运行时由任务管理器负责传入接口参数，模拟环境下需要调试者自行传入模拟值。

        参数说明:
            id: 当前节点 ID
            task_id: 当前任务 ID
            is_initiator: 当前节点是否是任务发起方
            recover: 是否使用恢复模式启动
        """
        # 先记录传入的参数，由于不支持恢复模式，可以忽略 recover
        self.id = id
        self.task_id = task_id
        self.is_initiator = is_initiator

        # 发起方作为聚合方，其它节点作为参与方
        if self.is_initiator:
            self._run_as_aggregator()
        else:
            self._run_as_collaborator()

    @abstractmethod
    def before_check_in(self, is_aggregator: bool):
        """完成集合前初始化本地资源。"""

    @abstractmethod
    def before_training(self, is_aggregator: bool):
        """训练开始前的初始化工作。"""

    @property
    @abstractmethod
    def model(self) -> Module:
        """获取训练使用的模型对象。"""

    @abstractmethod
    def train_an_epoch(self):
        """完成一个 epoch 训练的逻辑。"""

    @abstractmethod
    def run_test(self, writer: SummaryWriter):
        """测试的逻辑。"""

    def _run_as_aggregator(self):
        """作为聚合方运行."""
        # 初始化本地资源
        self.contractor = SimpleFedAvgContractor(task_id=self.task_id)
        self.data_channel = SharedFileDataChannel(contractor=self.contractor)

        self.collaborators = self.contractor.query_nodes()
        self.collaborators.remove(self.id)  # 把自己移出去
        self.checked_in = set()  # 记录集合的参与方
        self.result_dir = get_result_dir(self.task_id)
        self.log_dir = os.path.join(self.result_dir, 'tb_logs')  # 记录测试评估指标的目录
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)  # 记录测试评估指标的 writter

        self.round = 0

        # 调用 before_check_in 执行用户自定义的额外初始化逻辑。
        # 聚合方与参与方的初始化逻辑可能会不一样，所以加一个 is_aggregator 参数已做区分。接口定义也据此更新。
        self.before_check_in(is_aggregator=True)
        self.push_log(f'节点 {self.id} 初始化完毕。')

        # 监听集合请求
        self.push_log('开始等待参与成员集合 ...')
        for _event in self.contractor.contract_events():
            if isinstance(_event, CheckInEvent):
                # 收到集合请求后，记录参与者，并向请求方回复集合响应
                self.checked_in.add(_event.peer_id)
                self.contractor.response_check_in(aggr_id=self.id, peer_id=_event.peer_id)
                self.push_log(f'成员 {_event.peer_id} 加入。')
                # 统计参与者数量，如果集合完成退出循环
                if len(self.collaborators) == len(self.checked_in):
                    break  # 退出监听循环
        self.push_log(f'参与成员集合完毕，共有 {len(self.checked_in)} 位参与者。')

        # 完成训练初始化工作
        self.model  # 初始化模型
        # 调用 before_training 执行用户自定义的额外初始化逻辑。
        # 聚合方与参与方的初始化逻辑可能会不一样，所以加一个 is_aggregator 参数已做区分。接口定义也据此更新。
        self.before_training(is_aggregator=True)
        self.push_log(f'节点 {self.id} 准备就绪，可以开始执行计算任务。')

        for _round in range(self.rounds):
            # 发送开始训练的通知
            self.round = _round
            self.contractor.start()
            self.push_log(f'第 {_round + 1} 轮训练开始。')
            # 计数训练轮次，发送全局模型
            with TemporaryFile() as f:
                torch.save(self.model.state_dict(), f)
                f.seek(0)
                self.push_log('开始发送全局模型 ...')
                self.data_channel.batch_send_stream(source=self.id,
                                                    target=self.collaborators,
                                                    data_stream=f.read(),
                                                    ensure_all_succ=True)
            self.push_log('发送全局模型完成。')
            # 监听本地模型传输请求，收集本地模型
            self.updates = []  # 记录本地模型参数更新
            self.push_log('开始等待收集本地模型 ...')
            training_results = self.data_channel.batch_receive_stream(
                receiver=self.id,
                source_list=self.collaborators,
                ensure_all_succ=True
            )
            for _source, _result in training_results.items():
                buffer = io.BytesIO(_result)
                state_dict = torch.load(buffer)
                self.updates.append(state_dict)
                self.push_log(f'收到来自 {_source} 的本地模型。')
            # 聚合得到新的全局模型
            self.push_log('开始执行参数聚合 ...')
            self._make_aggregation()
            self.push_log('参数聚合完成。')
            # 如果达到训练轮次，循环结束

            # 使用测试数据做一次测试，得到全局模型的评估指标数据
            # 测试时指定 TensorBoard 的 writter，否则用户使用自定义的 writter，无法控制日志文件目录。
            # 接口定义也据此更新。
            self.push_log('训练完成，测试训练效果 ...')
            self.run_test(writer=self.tb_writer)
            self.push_log('测试完成。')

        # 上传模型文件和评估指标数据
        # 打包记录测试时写入的所有 TensorBoard 日志文件
        self.push_log('整理计算结果，准备上传 ...')
        report_file = os.path.join(self.result_dir, "report.zip")
        with ZipFile(report_file, 'w') as report_zip:
            for path, _, filenames in os.walk(self.log_dir):
                rel_dir = os.path.relpath(path=path, start=self.result_dir)
                rel_dir = rel_dir.lstrip('.')  # ./file => file
                for _file in filenames:
                    rel_path = os.path.join(rel_dir, _file)
                    report_zip.write(os.path.join(path, _file), rel_path)
        report_file_path = os.path.abspath(report_file)
        # 记录训练后的模型参数
        model_file = os.path.join(self.result_dir, "model.pt")
        with open(model_file, 'wb') as f:
            torch.save(self.model.state_dict(), f)
        model_file_path = os.path.abspath(model_file)
        # 调用接口执行上传
        self.contractor.upload_task_achivement(aggregator=self.contractor.EVERYONE[0],
                                               report_file=report_file_path,
                                               model_file=model_file_path)
        self.push_log('计算结果上传完成。')

        # 通知任务管理器训练结束，完成训练退出
        self.contractor.notify_task_completion(result=True)
        self.contractor.close()
        self.push_log('计算任务完成。')

    def _make_aggregation(self):
        """执行参数聚合。"""
        # 模型参数清零
        global_params = self.model.state_dict()
        for _param in global_params.values():
            if isinstance(_param, torch.Tensor):
                _param.zero_()
        # 累加收到的本地模型参数
        for _update in self.updates:
            for _key in global_params.keys():
                global_params[_key].add_(_update[_key])
        # 求平均值获得新的全局参数
        count = len(self.collaborators)
        for _key in global_params.keys():
            if global_params[_key].dtype in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
            ):
                global_params[_key].div_(count, rounding_mode='trunc')
            else:
                global_params[_key].div_(count)
        self.model.load_state_dict(global_params)

    def _run_as_collaborator(self):
        """作为参与方运行。"""
        # 初始化本地资源
        self.contractor = SimpleFedAvgContractor(task_id=self.task_id)
        self.data_channel = SharedFileDataChannel(contractor=self.contractor)

        # 调用 before_check_in 执行用户自定义的额外初始化逻辑。
        # 聚合方与参与方的初始化逻辑可能会不一样，所以加一个 is_aggregator 参数已做区分。接口定义也据此更新。
        self.before_check_in(is_aggregator=False)
        self.push_log(f'节点 {self.id} 初始化完毕。')

        # 广播发送集合请求
        self.push_log('发送集合请求，等待聚合方响应。')
        self.contractor.check_in(peer_id=self.id)
        # 监听集合响应，收到集合响应后，记录聚合方
        for _event in self.contractor.contract_events():
            if isinstance(_event, CheckInResponseEvent):
                self.aggregator = _event.aggr_id
                self.push_log('收到响应，集合成功。')
                break  # 退出监听循环

        # 完成训练初始化工作
        self.model  # 初始化模型
        # 调用 before_training 执行用户自定义的额外初始化逻辑。
        # 聚合方与参与方的初始化逻辑可能会不一样，所以加一个 is_aggregator 参数已做区分。接口定义也据此更新。
        self.before_training(is_aggregator=False)
        self.push_log(f'节点 {self.id} 准备就绪，可以开始执行计算任务。')

        while True:
            self.push_log('等待训练开始信号 ...')
            # 监听开始训练消息或训练结束消息；如果收到开始训练消息，跳到第 6 步；如果收到训练结束消息，完成训练退出
            for _event in self.contractor.contract_events():
                if isinstance(_event, StartEvent):
                    self.push_log('开始训练 ...')
                    break  # 退出监听循环
                elif isinstance(_event, CloseEvent):
                    self.push_log('训练完成。')
                    return  # 退出训练
            # 监听传输全局模型消息
            self.push_log('等待接收全局模型 ...')
            _, data_stream = self.data_channel.receive_stream(receiver=self.id,
                                                              source=self.aggregator)
            buffer = io.BytesIO(data_stream)
            new_state = torch.load(buffer)
            self.model.load_state_dict(new_state)
            self.push_log('接收全局模型成功。')
            # 在本地做一个 epoch 的训练，更新本地模型
            self.push_log('开始训练本地模型 ...')
            self.train_an_epoch()
            self.push_log('训练本地模型完成。')
            # 将本地模型发送给聚合方
            with TemporaryFile() as f:
                torch.save(self.model.state_dict(), f)
                f.seek(0)
                self.push_log('准备发送本地模型 ...')
                self.data_channel.send_stream(source=self.id,
                                              target=self.aggregator,
                                              data_stream=f.read())
                self.push_log('发送本地模型完成。')
            # 继续循环
