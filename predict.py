import torch
from torch.nn import Softmax, Sigmoid

import data.data_utils as dt
import params
import utils as ut
from lightning_module.sp_module import SPModule


def prepare_model():
    return {
        1: "cnn-aa-default-0_epochs=100",
        # "transformer-aa-lite-0_epochs=100",
        # "transformer-aa-lite-1_epochs=100",
        2: "transformer-aa-lite-0_epochs=100",
        3: "lstm-aa-default-0_epochs=100",
        4: "st_bilstm-aa-default-0_epochs=100",
        5: "bert-aa-default-0_epochs=100",
        6: "bert_pretrained-aa-default-0_epochs=100",
        7: "bert_pretrained-aa-default-0_epochs=100_v1",
        8: "cnn_trans-aa-lite-0_epochs=100",
        9: "cnn-smiles-default-0_epochs=100",
        10: "transformer-smiles-lite-0_epochs=100",
        11: "cnn_trans-smiles-lite-0_epochs=100",
        12: "gconv-graph-heavy-0_epochs=100",
        13: "gconv_trans-graph-default-0_epochs=100",
        14: "cnn-aa-default-1_epochs=100",
        15: "transformer-aa-lite-1_epochs=100",
        16: "lstm-aa-lite-1_epochs=100",
        17: "st_bilstm-aa-lite-1_epochs=100",
        18: "bert-aa-default-1_epochs=100",
        19: "bert_pretrained-aa-default-1_epochs=100",
        20: "bert_pretrained-aa-default-1_epochs=100_v1",
        21: "cnn_trans-aa-lite-1_epochs=100",
        22: "cnn-smiles-default-1_epochs=100",
        23: "transformer-smiles-lite-1_epochs=100",
        24: "cnn_trans-smiles-lite-1_epochs=100",
        25: "gconv-graph-heavy-1_epochs=100",
        26: "gconv_trans-graph-default-1_epochs=100",
    }


def prepare_org():
    return {
        0: "Eukaryotes",
        1: "Gram-negative bacteria",
        2: "Gram-positive bacteria",
        3: "Archaea",
    }


def _extract_metric_filename(filename):
    """Extract info from metric filename: model, data, used_org, organism
    """
    model_name, data, conf = filename.split('-')[0:3]
    use_org = int(filename.split("-")[3].split('_')[0])
    if '_v1' in filename:
        model_name = model_name + '_freeze'

    return model_name, data, conf, use_org


if __name__ == '__main__':
    models = prepare_model()
    # print(models[9])
    protein_string = input("Input protein string (AA Seq): ")
    # protein_string = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL"
    for k, model_type in models.items():
        print(f'{k}: {model_type}')
    mtype = int(input("Input model type: "))
    model = models[mtype]

    model_name, data, conf, use_org = _extract_metric_filename(model)
    if data == 'smiles':
        protein_string = dt.get_smiles_of_prot(protein_string)

    checkpoint = ut.abspath(f'checkpoints/{model}.ckpt')
    sp_module = SPModule.load_from_checkpoint(checkpoint)
    # protein_string = torch.tensor(protein_string, device='cuda')
    # softmax = Softmax(dim=1)
    softmax = Sigmoid()
    pred = None
    if use_org:
        orgs = prepare_org()
        for k, org in orgs.items():
            print(f'{k}: {org}')
        otype = int(input("Choose Organism: "))
        org = orgs[otype]
        pred = softmax(sp_module([protein_string], [org]))
    else:
        pred = softmax(sp_module([protein_string]))
    print(pred)
    label = torch.argmax(pred, dim=1).item()
    for k, lb in params.SP_LABELS.items():
        if lb == label:
            print(f'{k}')
