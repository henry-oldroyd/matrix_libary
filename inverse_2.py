from matricies import Matrix
from pprint import pprint

def gaussian_elimination(matrix: Matrix):
    # https://www.youtube.com/watch?v=vKBNzM3V-Rc
    assert matrix.is_square()
    n = len(matrix.matrix_data)
    identity_data = [[0]*i + [1] + [0]*(n-(i+1)) for i in range(0, n)]
    # print(identity_data)
    identity = Matrix(identity_data)

    augmented_matrix = [list(m_row) + i_row for m_row, i_row in zip(matrix.rows(), identity.rows())]
    print(augmented_matrix)

    for pivot_num in range(n):
        pivot = augmented_matrix[pivot_num][pivot_num]
        # print(pivot)

        other_rows_index = list(range(n))
        other_rows_index.pop(pivot_num)

        for other_row_index in other_rows_index:
            value_in_pivot_column = augmented_matrix[other_row_index][pivot_num]
            pivot_rows_to_add = -value_in_pivot_column / pivot

            for index in range(2*n):
                augmented_matrix[other_row_index][index] += pivot_rows_to_add * augmented_matrix[pivot_num][index]

        for index in range(2*n):
            augmented_matrix[pivot_num][index] = (1/pivot) * augmented_matrix[pivot_num][index]

    # remove other data 

    print(augmented_matrix)
    for i in range(n):
        augmented_matrix[i] = augmented_matrix[i][n:]

    print(augmented_matrix)
    inverse = Matrix(
        augmented_matrix
    )

    return inverse
if __name__ == "__main__":
    # print(
    (
        gaussian_elimination(
            Matrix(
                [
                    [-3, 2, -1],
                    [6, -6, 7],
                    [3, -4, 4]
                ]
            )
        )
    )