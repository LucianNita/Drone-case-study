import os, sys, json
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from multi_uav_planner.dubins import cs_segments_shortest, csc_segments_shortest
from multi_uav_planner.path_model import LineSegment, CurveSegment, Path

def path_to_dict(path: Path):
    out = {"segments": [], "length": path.length()}
    for s in path.segments:
        if isinstance(s, LineSegment):
            out["segments"].append({"type":"line","start":s.start,"end":s.end})
        elif isinstance(s, CurveSegment):
            out["segments"].append({"type":"arc","center":s.center,"radius":s.radius,"theta_s":s.theta_s,"d_theta":s.d_theta})
    return out

def main():
    cases = [
        {"kind":"CS", "start":(0,0,0.4), "end":(120,60), "R":30.0},
        {"kind":"CSC","start":(10,10,0.2),"end":(200,120,-0.7),"R":40.0},
    ]
    export = []
    for c in cases:
        if c["kind"] == "CS":
            p = cs_segments_shortest(c["start"], c["end"], c["R"])
        else:
            p = csc_segments_shortest(c["start"], c["end"], c["R"])
        export.append({"case":c, "path":path_to_dict(p)})
    with open("paths.json","w") as f:
        json.dump(export, f, indent=2)
    print("Exported to paths.json")

if __name__ == "__main__":
    main()