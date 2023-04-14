# 在参与方之间传输数据

由于前述的[消息传递机制](./message.md)依托于区块链网络，因此不适合传输较大体量的数据。当需要传输大体量数据，比如模型参数时，应当使用平台提供的数据传输工具。由于数据量往往比较大，点对点式的传输方式效率低下，难以满足性能需求，所以当前的平台实现借助共享文件的方式传输数据。数据发送方将需要发送的数据写入共享文件，数据接收方读取文件获得数据。由于读文件操作可以并行执行，可以有效提高数据传输的效率。

当前版本的数据传输工具需要配合合约消息通信机制共同工作，其中合约消息用于同步各方状态，传递关键参数。数据发送方将数据写入共享文件后，将共享文件链接地址发送至所有数据接收方。数据接收方收到通知后自行读取接收数据。

要实现数据传输需要三个步骤：初始化 `DataChannel` 对象、发送数据流、接收数据流。

## 初始化`DataChannel`对象

当前版本平台推荐的 `DataChannel` 实现是 `SharedFileDataChannel`。初始化 `DataChannel` 对象时需要传入当前任务使用的合约消息管理对象，否则 `DataChannel` 不能正常接收控制消息。而由于合约消息工具需要指定所属计算任务的 ID，所以 `DataChannel` 对象的初始化也需要在得到计算任务 ID 后才可以执行。

```Python
from alphafed.data_channel import SharedFileDataChannel
from alphafed.fed_avg.contractor import FedAvgContractor

...

contractor = FedAvgContractor(task_id=task_id)
data_channel = SharedFileDataChannel(self.contractor)
```

## 发送数据流

数据传输时发送的数据为 bytes 类型的字节流，因此所有数据在发送前需要转化为 bytes 流形式。发送接口为 `send_stream` 和 `batch_send_stream`，调用接口返回的是成功接收到数据的参与方 ID 列表。

```Python
buffer = io.BytesIO()
torch.save(self.state_dict(), buffer)
targets = [NODE_ID_LIST]
received = self.data_channel.send_stream(source=self.id,
                                         target=NODE_ID,
                                         data_stream=buffer.getvalue())
accept_list = self.data_channel.batch_send_stream(source=self.id,
                                                  target=targets,
                                                  data_stream=buffer.getvalue())
```

## 接收数据流

同样，接收方收到的数据也是 bytes 流数据。如果需要重新结构化为其它类型，比如 Python 对象类型，需要接收方自行处理。接收数据前，需要先监听发送方发送的数据传输消息，传输消息中包含接收数据的所有参数。将传输消息传入 `receive_stream` 接口后，接口会返回发送方 ID 和接收到的数据流信息。

```Python
# 接收数据前通过合约消息与发送方取得联系，同步状态
source_id, parameters = self.data_channel.receive_stream(receiver=self.id, source=SOURCE_ID)
buffer = io.BytesIO(parameters)
new_state_dict = torch.load(buffer)
```

如果需要从多个发送方同时接收数据，比如多个参与方向聚合方发送本地参数的场景，可以使用 `batch_receive_stream` 接口。发送完成后会返回发送方 ID 和发送数据流组成的映射字典。

```Python
# 接收数据前通过合约消息与发送方取得联系，同步状态
data_info = self.data_channel.batch_receive_stream(receiver=self.id,
                                                   source_list=[SOURCE_ID])
for sender, data in data_info.items():
    print(f'receive data from {sender}')
    buffer = io.BytesIO(parameters)
    new_state_dict = torch.load(buffer)
```
