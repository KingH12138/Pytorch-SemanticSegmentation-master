import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

# pil = Image.open(r'D:\PythonCode\Pytorch-SemanticSegmentation-master\demo_labelme\test_json\label.png')
# pil = np.array(pil)


outputs = torch.randint(0,4,(16,1,224,224))
outputs = F.one_hot(outputs, num_classes=5)
outputs = outputs.permute(0,4,2,3,1).squeeze(-1)
print(outputs.shape)
