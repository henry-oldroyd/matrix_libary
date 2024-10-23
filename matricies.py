import itertools
from typing import Union
import math

dim_type = dict[str, int]
matrix_array_type = list[list[Union[float, int]]]

def create_empty_list(a: int, b: int) -> list[list[None]]:
    """crates a 2d array with a rows and b columns. It is filled with None. a>=1, b>=1"""
    # type checks
    try:
        assert isinstance(a, int), "a must be of type int"
        assert isinstance(b, int), "b must be of type int"
    except AssertionError as e:
        raise TypeError(e) from e
    # value checks
    try:
        assert 1 <= a, "a must be greater than 1"
        assert 1 <= b, "b must be greater than 1"
    except AssertionError as e:
        raise ValueError(e) from e

    return [[None for _ in range(b)] for _ in range(a)]


class Matrix:
    def __init__(self, matrix_array: matrix_array_type):
        # type checks
        try:
            assert isinstance(matrix_array, list), "matrix_array must be of type list"
            assert all(isinstance(row, list) for row in matrix_array), "rows must be of type list"
            def gen():
                for row in matrix_array:
                    yield from row
            assert all(isinstance(item, (int, float)) for item in gen()), "all items must be floats or int"
        except AssertionError as e:
            raise TypeError(e) from e

        no_rows = len(matrix_array)
        no_cols = len(matrix_array[0])
        try:
            # ensure non zero rows and cols
            assert no_rows >= 1, "Must be 1 or more rows"
            assert no_cols >= 1, "Must be 1 or more cols"
            # ensure cols are consistent, assert all row lengths consistent
            assert all(len(row) == no_cols for row in matrix_array), "Length of all rows bust be consistent"
        except AssertionError as e:
            raise ValueError(e) from e

        # assign parameters as attributes
        self.matrix_data = matrix_array
        self.no_rows, self.no_cols = no_rows, no_cols
    
    def __eq__(self, m):
        # if not of type matrix
        if not isinstance(m, Matrix): return False
        # if not the same dimension
        if not (m.no_rows, m.no_cols) == (self.no_rows, self.no_cols): return False

        rows, cols = self.no_rows, self.no_cols
        return all(math.isclose(self.matrix_data[i][j], m.matrix_data[i][j]) for i, j in itertools.product(range(rows), range(cols)))

    def __repr__(self):
        return self.matrix_data
    
    def __str__(self) -> str:
        return str(self.matrix_data)

    def is_square(self):
        return self.no_cols == self.no_rows

    def all_items(self):
        for row in self.matrix_data:
            yield from row
    
    def item(self, row, col) -> Union[float, int]:
        return self.matrix_data[row][col]
    
    def rows(self):
        yield from self.matrix_data
    
    def row(self, i: int):
        return self.matrix_data[i]
    
    def columns(self):
        for i in range(self.no_cols):
            yield [row[i] for row in self.matrix_data]

    def column(self, i: int):
        return [row[i] for row in self.matrix_data]



    def scalar_multiple(self, scalar: Union[float, int]):
        rows, cols = self.no_rows, self.no_cols
        # for i in range(rows):
        #     for j in range(cols):
        #         self.matrix_data[i][j] *= scalar

        # https://www.geeksforgeeks.org/python-itertools-product/
        for i, j in itertools.product(range(rows), range(cols)):
            self.matrix_data[i][j] *= scalar

    def sub_matrix(self, row_ex: int, col_ex: int) -> matrix_array_type:        
        # type validation
        try:
            assert all(isinstance(x, int) for x in (row_ex, col_ex)), "Both row and col must be an int"
        except AssertionError as e:
            raise TypeError(e) from e


        # value validation
        try:
            assert 0 <= row_ex <= self.no_rows, "row given out of range"
            assert 0 <= col_ex <= self.no_cols, "col given out of range"
        except AssertionError as e:
            raise ValueError(e) from e

        result_rows: matrix_array_type = create_empty_list(self.no_rows-1, self.no_cols-1)

        for row_i in range(self.no_rows):
            if row_i == row_ex:
                continue
            for col_i in range(self.no_cols):
                if col_i == col_ex:
                    continue

                out_row_i = row_i
                out_col_i = col_i

                if out_row_i > row_ex:
                    out_row_i += -1
                if out_col_i > col_ex:
                    out_col_i += -1

                # print(f"temp = self.matrix_data[{row_i}][{col_i}]")
                # temp = self.matrix_data[row_i][col_i]
                # print(f"temp -->  {temp}")
                # print(f"assigning to result_rows[{out_row_i}][{out_col_i}]")
                # result_rows[out_row_i][out_col_i] = temp

                result_rows[out_row_i][out_col_i] = self.matrix_data[row_i][col_i]
        return result_rows
    
    
    def det(self) -> int:
        if not self.is_square():
            raise ValueError("This matrix is not square so has no determinant")
        
        # base case
        if self.no_rows == 2:
            a,b,c,d = list(self.all_items())
            return a*d - b*c

        # recursive case
        det_m = 0
        first_row = list(self.rows())[0]

        # rows, cols = self.no_rows, self.no_cols
        for i, item in enumerate(first_row):
            scalar = item * (-1)**(i % 2)
        
            sub_matrix_data: matrix_array_type = self.sub_matrix(0, i)
            sub_matrix: Matrix = Matrix(matrix_array= sub_matrix_data)
            # recursive call
            det_m += scalar * sub_matrix.det()

        return det_m


    @staticmethod
    def dot_product(v1: list, v2: list) -> Union[float, int]:
        # type validation
        try:
            for v in (v1, v2):
                assert isinstance(v, list), "Both vestors be of type list"
                assert all(isinstance(e, (float, int)) for e in v), "Vectors must only contain items of type int of float"
        except AssertionError as e:
            raise TypeError(e) from e

        # value validation
        try:
            assert len(v1) == len(v2), "The number of rows in each vector mush match"
            # assert len(v1) >= 1, "The vectors must have a non-zero number of rows"
            assert v1, "The vectors must have a non-zero number of rows"
        except AssertionError as e:
            raise ValueError(e) from e

        return sum(val1 * val2 for val1, val2 in zip(v1, v2))

    @classmethod
    def matrix_product(cls, m1, m2):
        # validation
        try:
            assert all(isinstance(m, cls) for m in (m1, m2)), "m1 and m2 must be of type matrix"

        except AssertionError as e:
            raise TypeError(e) from e
        try:
            assert m1.no_cols == m2.no_rows, "m1 column num but equal number of m2 rows to multiply"
        except AssertionError as e:
            raise ValueError(e) from e

        # r is for result
        r_rows, r_cols = m1.no_rows, m2.no_cols
        r_array: list[list[None]] = create_empty_list(r_rows, r_cols)

        for i in range(r_rows):
            for j in range(r_cols):
                v1 = m1.row(i)
                v2 = m2.column(j)
                r_array[i][j] = cls.dot_product(v1=v1, v2=v2)
        return cls(r_array)

    def __mul__(self, x):
        if not isinstance(x, (int, float, Matrix)):
            raise TypeError("Matrix multiplication only supported with another matrix or a scalar (int/float)")

        if isinstance(x, Matrix):
            self = self.matrix_product(self, x)
        else:
            self.scalar_multiple(x)
    
    def adjugate(self):
        if not self.is_square():
            raise ValueError("This matrix is not square so has no determinant")

        # https://semath.info/src/inverse-cofactor-ex4.html
        # adj[i, j] = (-1)^(i+j) * M[i, j]

        # get adjugate
        adj_data = create_empty_list(self.no_rows, self.no_cols)
        for i, j in itertools.product(range(self.no_rows), range(self.no_cols)):
            adj_data[i][j] = (-1)*((i+j) % 2) * self.matrix_data[i][j]

        return Matrix(adj_data)


    # # FACTORIAL SQUARED TIME COMPLEXITY!!!
    # def inverse(self):
    #     det = self.det()
    #     if math.isclose(det, 0):
    #         raise ValueError("This matrix is singular: det = 0, therefore no inverse")

    #     adj = self.adjugate()
    #     adj.scalar_multiple(1/det)
    #     return adj
    
    ## IMPROVEMENT WITH GAUSSIAN ELIMINATION
    ## https://www.khanacademy.org/math/algebra-home/alg-matrices/alg-determinants-and-inverses-of-large-matrices/v/inverting-matrices-part-3
    
    def inverse(self):
        # get elementary row operation to turn m into i, apply them again to turn i into m^-1
        if not self.is_square():
            raise ValueError("Matrix must be square to inverse")
        n = self.no_rows
        
        identify_matrix = Matrix([0]*i + [1] + [0]*n-(i+1) for i in range(1, n+1))
        print(identify_matrix)
    

def test():
    m2: Matrix = Matrix([
        [9,4,3],
        [1,8,6],
        [11,5,3]
    ])
    assert m2.det() == -51
    assert m2.no_rows, m2.no_cols == (3,3)

    m3: Matrix = Matrix([
        [4, 3, 2, 2],
        [0, 1, -3, 3],
        [0, -1, 3, 3],
        [0, 3, 1, 1]
    ])
    assert m3.det() == -240
    assert m3.no_rows, m3.no_cols == (4,4)

    m4 = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    m5 = Matrix([
        [7, 8],
        [9, 10],
        [11, 12]
    ])
    
    m6 = Matrix([
        [58, 64],
        [139, 154]
    ])
    
    assert Matrix.matrix_product(m4, m5) == m6
    print(Matrix.matrix_product(m4, m5))
    
    m7 = Matrix([
        [5, 22, 33, 7],
        [33, 3, 4, 35],
        [7, 77, 3345, -9],
        [0, 53, 0, 5]
    ])

    assert m7 == m7

    m8 = Matrix([
        [1, 2, -1],
        [2, 1, 2],
        [-1, 2, 1]
    ])

    m9 = Matrix([
        [-3, -4, 5],
        [-4, 0, -4],
        [5, -4, -3]
    ])
    
    print(m8.adjugate())
    assert m8.adjugate() == m9
    

if __name__ == '__main__':
    test()
    # pass