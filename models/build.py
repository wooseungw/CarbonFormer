from models.unet import  UNet_carbon




def build_model(net, channel, dropout=False):


    if net == "UNet_carbon":
        # class_num = channel
        model = UNet_carbon(num_classes=channel, dropout=dropout)

    return model
    