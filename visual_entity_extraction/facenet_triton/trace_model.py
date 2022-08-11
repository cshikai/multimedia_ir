from facenet_pytorch import InceptionResnetV1
import torch

# Create an inception resnet (in eval mode):
print("Loading model...")
model = InceptionResnetV1(pretrained='vggface2').cuda()
model.eval()

# You may either generate random tensor (with the appropriate dimensions), or load an example data from your dataset
example = torch.rand(1, 3, 224, 224).cuda() # Example of an image of pixel 224x224, 3 channels, batch size of 1.

# Trace model using example tensor
print("Tracing model...")
traced_script_module = torch.jit.trace(model, example)

# Verify that output of traced model is identical to original model
print("Verifying traced model...")
orig_output = model(example)
traced_output = traced_script_module(example)
if torch.all(orig_output.eq(traced_output)):
    traced_script_module.save("model.pt")
    print("Model saved successfully as model.pt")
else:
    raise Exception('ERROR: Output traced model is not identitcal to original model.')    

