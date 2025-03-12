import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

processor = Owlv2Processor.from_pretrained("owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("owlv2-base-patch16-ensemble")

img_path = "bottle.jpg"
image = Image.open(img_path)
texts = [[" detect plastic bottle"]]
inputs = processor(text=texts, images=image, return_tensors="pt")

input_ids=inputs['input_ids']#.type(torch.int64)
attention_mask=inputs['attention_mask']#.type(torch.int64)


pixel_values=inputs['pixel_values']


print('input_ids:',input_ids.shape)
print('attention_mask:',attention_mask.shape)
print('pixel_values:',pixel_values.shape)

# trace_model = torch.jit.trace(model,data_inputs)
# trace_model.save("/home/yanghong/project/CLIPSeg/clipseg/weights/owlvit_trace.pt")
onnx_path="./weights/owlv2_all.onnx"
torch.onnx.export(
    model,
    f=onnx_path,
    args=(input_ids,pixel_values,attention_mask), #, zeros, ones),
    input_names=["attention_mask","pixel_values", "input_ids"],
    output_names=["logits","text_embeds","pred_boxes","image_embeds"],
    opset_version=17,
    do_constant_folding=True,
    verbose=True,
    )


print("export onnx model success")

'''
optimum-cli export onnx --model ./owlv2-base-patch16-ensemble --task zero-shot-object-detection ./weights/

optimum-cli export onnx --model ./owlv2-base-patch16-ensemble  --task zero-shot-object-detection    ./weights/
'''