import onnx
from onnxsim import simplify
import numpy as np

model = onnx.load("model_1.onnx")
graph = model.graph
for node in graph.node:
    if node.op_type == "LSTM":
        new_input = node.input
        new_input[4] = node.name + '_seq_lens'
        node.input.remove(node.input[-1])
        # new_input[-1] = node.name + '_P'
        seq_lens = onnx.helper.make_tensor(new_input[4], onnx.TensorProto.INT32, [1], np.ones(1, dtype=np.int32))
        hidden_size = 128
        # P = onnx.helper.make_tensor(new_input[-1], onnx.TensorProto.FLOAT, [1, 3 * hidden_size], np.zeros((1, 3 * hidden_size), dtype=np.float32))
        graph.initializer.append(seq_lens)
        # graph.initializer.append(P)

        for i, attr in enumerate(node.attribute):
            if attr.name == "activations":
                del node.attribute[i]
onnx.checker.check_model(model)
sim_model, _ = simplify(model)
onnx.save(sim_model, "modified_model_1.onnx")                