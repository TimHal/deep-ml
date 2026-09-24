import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	
    a = np.array(a)
    b = np.array(b)

    dim_target = (a.shape[0], b.shape[1])
    res = np.empty(dim_target)

    # Make sure the dimensions are compatible.
    # In the loop we will also assert that each row/column matches
    # the respective dimension as expected here.

    if a.shape[1] != b.shape[0]:
        return -1

    for i in range(dim_target[0]):
        for j in range(dim_target[1]):

            row = a[i]
            col = b[:,j]

            if len(row) != len(col):
                return -1

            x = [row[k] * col[k] for k in range(len(row))]
            res[i][j] = sum(x)

    return res
            