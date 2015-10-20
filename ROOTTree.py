from ROOT import *

if __name__ == "__main__":
    tree = TTree('Tree','')
    tree.ReadFile('../Data/train1.csv',"",',')
    outfile = TFile.Open('outtree.root','recreate')
    tree.Write()
    outfile.Close()
    
    
