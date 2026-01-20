import argparse
import os
import json
from JetIDHelpers import create_jetId_correctionSet, write_correctionSet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create correctionlib-compatible JSON from JetID spec JSON file")
    parser.add_argument("jetID_spec_json_file", type=str, help="JetID spec JSON file")
    parser.add_argument("--description", "-d", type=str, required=False, default="Jet Identification Criteria", help="Overall description of JetID correctionSet", dest="description")
    parser.add_argument("--input_abs_eta", action="store_true", default=False, help="whether to use eta or abs(eta) as input", dest="input_abs_eta")
    parser.add_argument("--out_filename", "-o", type=str, required=False, default="jetID.json", help="Filename of output JSON", dest="out_filename")
    parser.add_argument("--write_gzip", action="store_true", default=False, help="whether to write .json.gz file in addition to .json file", dest="write_gzip")
    parser.add_argument("--write_summary", action="store_true", default=False, help="whether to write html summary file in from generated JSON file", dest="write_summary")
    args = parser.parse_args()

    jetId_tasks = list()
    with open(args.jetID_spec_json_file) as f:
        jetId_tasks = json.load(f)
    
    corrSet = create_jetId_correctionSet(jetId_tasks, description=args.description, input_abs_eta=args.input_abs_eta)
    write_correctionSet(corrSet, args.out_filename, args.write_gzip)
    if args.write_summary:
        os.system("correction --html {} summary {}".format(os.path.splitext(args.out_filename)[0]+".html", args.out_filename))
