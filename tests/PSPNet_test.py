from models import PSPNet

if __name__ == "__main__":
    model = PSPNet.get_pspnet(name="pspnet_s7")
    model.summary()