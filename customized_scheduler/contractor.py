import secrets
from dataclasses import dataclass
from typing import List

from alphafed.contractor.common import ContractEvent
from alphafed.contractor.task_message_contractor import (
    ApplySharedFileSendingDataEvent, TaskMessageContractor,
    TaskMessageEventFactory)


@dataclass
class CheckinEvent(ContractEvent):
    """An event of checkin for a specific task."""

    TYPE = 'checkin'

    peer_id: str
    nonce: str

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'CheckinEvent':
        event_type = contract.get('type')
        peer_id = contract.get('peer_id')
        nonce = contract.get('nonce')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        assert peer_id and isinstance(peer_id, str), f'invalid peer_id: {peer_id}'
        assert nonce or isinstance(nonce, str), f'invalid nonce: {nonce}'
        return CheckinEvent(peer_id=peer_id, nonce=nonce)


@dataclass
class CheckinResponseEvent(ContractEvent):
    """An event of responding checkin event."""

    TYPE = 'checkin_response'

    round: int
    aggregator: str
    nonce: str

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'CheckinResponseEvent':
        event_type = contract.get('type')
        round = contract.get('round')
        aggregator = contract.get('aggregator')
        nonce = contract.get('nonce')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        assert isinstance(round, int) and round >= 0, f'invalid round: {round}'
        assert aggregator and isinstance(aggregator, str), f'invalid aggregator: {aggregator}'
        assert nonce and isinstance(nonce, str), f'invalid nonce: {nonce}'
        return CheckinResponseEvent(round=round, aggregator=aggregator, nonce=nonce)


@dataclass
class StartRoundEvent(ContractEvent):
    """An event of starting a new round of training."""

    TYPE = 'start_round'

    round: int
    calculators: List[str]
    aggregator: str

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'StartRoundEvent':
        event_type = contract.get('type')
        round = contract.get('round')
        calculators = contract.get('calculators')
        aggregator = contract.get('aggregator')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        assert isinstance(round, int) and round > 0, f'invalid round: {round}'
        assert (
            calculators and isinstance(calculators, list)
            and all(_peer_id and isinstance(_peer_id, str) for _peer_id in calculators)
        ), f'invalid participants: {calculators}'
        assert aggregator and isinstance(aggregator, str), f'invalid aggregator: {aggregator}'
        return StartRoundEvent(round=round,
                               calculators=calculators,
                               aggregator=aggregator)


@dataclass
class ReadyForAggregationEvent(ContractEvent):
    """An event of notifying that the aggregator is ready for aggregation."""

    TYPE = 'ready_for_aggregation'

    round: int

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'ReadyForAggregationEvent':
        event_type = contract.get('type')
        round = contract.get('round')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        assert isinstance(round, int) and round > 0, f'invalid round: {round}'
        return ReadyForAggregationEvent(round=round)


@dataclass
class CloseRoundEvent(ContractEvent):
    """An event of closing a specific round of training."""

    TYPE = 'close_round'

    round: int

    @classmethod
    def contract_to_event(cls, contract: dict) -> 'CloseRoundEvent':
        event_type = contract.get('type')
        round = contract.get('round')
        assert event_type == cls.TYPE, f'合约类型错误: {event_type}'
        assert isinstance(round, int) and round > 0, f'invalid round: {round}'
        return CloseRoundEvent(round=round)


UploadTrainingResultsEvent = ApplySharedFileSendingDataEvent
DistributeParametersEvent = ApplySharedFileSendingDataEvent


class SimpleFedAvgEventFactory(TaskMessageEventFactory):

    _CLASS_MAP = {
        CheckinEvent.TYPE: CheckinEvent,
        CheckinResponseEvent.TYPE: CheckinResponseEvent,
        StartRoundEvent.TYPE: StartRoundEvent,
        ReadyForAggregationEvent.TYPE: ReadyForAggregationEvent,
        CloseRoundEvent.TYPE: CloseRoundEvent,
        **TaskMessageEventFactory._CLASS_MAP
    }


class SimpleFedAvgContractor(TaskMessageContractor):

    def __init__(self, task_id: str):
        super().__init__(task_id=task_id)
        self._event_factory = SimpleFedAvgEventFactory

    def checkin(self, peer_id: str) -> str:
        """Checkin to the task.

        :return
            A nonce string used for identifying matched sync_state reply.
        """
        nonce = secrets.token_hex(16)
        event = CheckinEvent(peer_id=peer_id, nonce=nonce)
        self._new_contract(targets=self.EVERYONE, event=event)
        return nonce

    def respond_check_in(self,
                         round: int,
                         aggregator: str,
                         nonce: str,
                         requester_id: str):
        """Respond checkin event."""
        event = CheckinResponseEvent(round=round, aggregator=aggregator, nonce=nonce)
        self._new_contract(targets=[requester_id], event=event)

    def start_round(self,
                    calculators: List[str],
                    round: int,
                    aggregator: str):
        """Create a round of training."""
        event = StartRoundEvent(round=round,
                                calculators=calculators,
                                aggregator=aggregator)
        self._new_contract(targets=self.EVERYONE, event=event)

    def notify_ready_for_aggregation(self, round: int):
        """Notify all that the aggregator is ready for aggregation."""
        event = ReadyForAggregationEvent(round=round)
        self._new_contract(targets=self.EVERYONE, event=event)

    def close_round(self, round: int):
        """Start a round of training."""
        event = CloseRoundEvent(round=round)
        self._new_contract(targets=self.EVERYONE, event=event)
