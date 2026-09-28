from rknn.api import RKNN

rknn = RKNN()

print('[1] loading onnx...')
rknn.load_onnx(model='behavior_lstm.onnx')

print('[2] building...')
rknn.build(do_quantization=True, dataset=None)

print('[3] exporting rknn...')
rknn.export_rknn('behavior_lstm.rknn')

print('[DONE]')
