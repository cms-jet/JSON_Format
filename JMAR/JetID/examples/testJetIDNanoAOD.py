# on lxplus, first load lcg stack
# e.g. source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

import argparse

import numpy as np

import uproot
import awkward as ak
import correctionlib

def computeJetID_direct(jets):
    abs_eta = np.abs(jets["eta"])
    chHEF = jets["chHEF"]
    neHEF = jets["neHEF"]
    chEmEF = jets["chEmEF"]
    neEmEF = jets["neEmEF"]
    muEF = jets["muEF"]
    chMultiplicity = jets["chMultiplicity"]
    neMultiplicity = jets["neMultiplicity"]
    multiplicity = chMultiplicity+neMultiplicity

    # Hardcode AK4Puppi JetID TightLeptonVeto Rereco2022CDE
    # from https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    mask0 = (abs_eta<=2.6) & (neHEF<0.99) & (neEmEF<0.90) & (multiplicity>1) & (muEF<0.80) & (chHEF>0.01) & (chMultiplicity>0) & (chEmEF<0.80)
    mask1 = (abs_eta>2.6) & (abs_eta<=2.7) & (neHEF<0.90) & (neEmEF<0.99) & (muEF<0.80) & (chEmEF<0.80)
    mask2 = (abs_eta>2.7) & (abs_eta<=3.0) & (neHEF<0.99)
    mask3 = (abs_eta>3.0) & (abs_eta<=5.0) & (neEmEF<0.4) & (neMultiplicity>=2)
    return mask0 | mask1 | mask2 | mask3

def computeJetID_correctionlib(jets, correction):
    eta = jets["eta"]
    chHEF = jets["chHEF"]
    neHEF = jets["neHEF"]
    chEmEF = jets["chEmEF"]
    neEmEF = jets["neEmEF"]
    muEF = jets["muEF"]
    chMultiplicity = jets["chMultiplicity"]
    neMultiplicity = jets["neMultiplicity"]
    multiplicity = chMultiplicity+neMultiplicity
    
    return correction.evaluate(eta, chHEF, neHEF, chEmEF, neEmEF, muEF, chMultiplicity, neMultiplicity, multiplicity)

if __name__ == "__main__":
    # parse file name
    parser = argparse.ArgumentParser(description="test JetID correctionlib JSON file")
    parser.add_argument("filename", type=str, help="input NanoAOD file")
    args = parser.parse_args()
    filename = args.filename
    
    # open with uproot
    file = uproot.open(filename)

    events = file["Events"].arrays(filter_name="/Jet*/i", how="zip")
    jets = events["Jet"]
    
    # select only jets with abs(eta) <= 5.0
    jets = jets[np.abs(jets["eta"])<=5.0]
    
    # direct
    jetID_direct = computeJetID_direct(jets)
    
    # with correctionlib
    jetID_filename = "../corrections/JetID_Run3_Rereco2022CDE_v2.json"
    evaluator = correctionlib.CorrectionSet.from_file(jetID_filename)
    correction_name = "AK4PUPPI_TightLeptonVeto"
    correction = evaluator[correction_name]

    counts = ak.num(jets)
    jets_flatten = ak.flatten(jets)
    jetID_correctionlib = ak.unflatten(computeJetID_correctionlib(jets_flatten, correction), counts=counts)

    # compute match
    num_jets = ak.sum(ak.num(jets))
    jetID_diff = jetID_correctionlib - jetID_direct
    num_match = ak.sum(jetID_diff == 0)
    num_mismatch = num_jets - num_match
    num_TP = int(ak.sum(jetID_correctionlib * jetID_direct)) # correctionlib = direct = True
    num_TN = int(num_match - num_TP) # correctionlib = direct = True
    num_FP = ak.sum(jetID_diff == 1) # correctionlib = True, direct = False
    num_FN = ak.sum(jetID_diff == -1) # correctionlib = False, direct = True
    
    metric_length = 35
    value_length = len(str(num_jets))
    print(f"{'number of jets:':<{metric_length}} {num_jets:<{value_length}}")
    print(f"{'number of match JetID:':<{metric_length}} {num_match:<{value_length}} ({num_match/num_jets*100:3.5f}%)")
    print(f"{'number of mismatch JetID:':<{metric_length}} {num_mismatch:<{value_length}} ({num_mismatch/num_jets*100:3.5f}%)")
    print()
    print(f"{'true positive  (both=1):':<{metric_length}} {num_TP:<{value_length}} ({num_TP/num_jets*100:3.5f}%)")
    print(f"{'true negative  (both=0):':<{metric_length}} {num_TN:<{value_length}} ({num_TN/num_jets*100:3.5f}%)")
    print(f"{'false positive (json=1, truth=0):':<{metric_length}} {num_FP:<{value_length}} ({num_FP/num_jets*100:3.5f}%)")
    print(f"{'false negative (json=0, truth=1):':<{metric_length}} {num_FN:<{value_length}} ({num_FN/num_jets*100:3.5f}%)")

    file.close()
