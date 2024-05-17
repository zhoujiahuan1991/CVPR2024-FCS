from models.fcs import FCS
def get_model(model_name, args):
    name = model_name.lower()


    if name == "fcs":
        return FCS(args)
    else:
        assert 0
