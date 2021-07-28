import numpy as np 
import json 

from autocali import reconstruct


def read_data():
    f = open('calibration.json')
    data = json.load(f)
    projections = [np.array(data['camera_{}'.format(i)]['projection']) for i in [1,2,3]]
    intrinsic   = [np.array(data['camera_{}'.format(i)]['intrinsic']) for i in [1,2,3]]
    f.close()
    return projections, intrinsic

def main():
    Ps, Ks = read_data()
    Q, ws = reconstruct.minimize_dual_quadric(Ps)
    """
    if Q != None:
        H = reconstruct.calculate_euclidean_homography(Q)
        Ks = reconstruct.calculate_Ks(ws)
        print('The euclidean homography: {}'.format(H))
    """


    



if __name__ == '__main__':
    main()








    
    