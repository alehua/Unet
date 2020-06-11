import argparse
import os
from solver import Solver
from data_loader import get_loader,MY_data
from PIL import Image
import platform
from torchvision import transforms as T

def main():
    sys = platform.system()
    if sys == "Windows":
        model_path   =  "models\\"
        result_path  = "result\\"
        train_path   = "image\\train"
        valid_path   = "dataset\\valid"
        test_path    = "dataset\\test"
    else:
        model_path   =  "./models"
        result_path  =  "./result"
        train_path   =  "./image/train"
        valid_path   =  "./dataset/valid"
        test_path    =  "./dataset/test"

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    
    train_loader = get_loader(image_path=train_path,image_size = 512,batch_size=6,num_workers = 0,mode='train',augmentation_prob=0.6)
  
    
    net = Solver( train_loader )
    # net.train()                
    net.test("models//5.pkl")     

if __name__ == '__main__':
    main()





# dataset = MY_data(train_path)
#     print(dataset.__len__())
#     a,b = dataset.__getitem__(3)
#     print(a.size())
#     new_img = T.ToPILImage()(b).convert('RGB')
#     new_img.show()

# dataiter = iter(train_loader)
#     image,label = next(dataiter)
#     print(type(label))
#     new_img = T.ToPILImage()(image[3]).convert('RGB')
#     new_img.show()