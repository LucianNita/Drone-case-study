import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from multi_uav_planner.dubins import cs_segments_shortest, csc_segments_shortest

def main():
    kind = input("Type [CS/CSC]: ").strip().upper()
    if kind != "CS" or kind != "CSC":
        raise ValueError("Error! Please insert either CS or CSC") 
    R = float(input("R: "))
    x0 = float(input("x0: ")); y0 = float(input("y0: ")); th0 = float(input("theta0 (rad): "))
    xf = float(input("xf: ")); yf = float(input("yf: "))
    if kind == "CS":
        p = cs_segments_shortest((x0,y0,th0),(xf,yf),R)
    else:
        thf = float(input("thetaf (rad): "))
        p = csc_segments_shortest((x0,y0,th0),(xf,yf,thf),R)
    print("Length:", p.length())

if __name__ == "__main__":
    main()