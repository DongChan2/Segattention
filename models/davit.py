import timm 
def Davit_tiny_Embedding(pretrained=True):
    return timm.create_model('davit_tiny.msft_in1k',pretrained=pretrained,num_classes=0)



if __name__=='__main__':
    import torch
    from torchsummaryX import summary
    model = Davit_tiny_Embedding()
    summary(model,torch.zeros(1,3,224,128))