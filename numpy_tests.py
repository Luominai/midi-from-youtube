import numpy as np

# # if I use array deconstruction to grab an int from an np array, then modify that int. Does that replicate to the array
# # Conclusion: No
# array = np.arange(3)
# [a, b, c] = array
# a += 1
# print(a)
# print(array) # 1 if replicated. 0 if not


# if I use array deconstruction to grab a subarray from an np array, then modify that subarray. Does that replicate to the array
# Conclusion: Yes
# array = np.arange(6).reshape(2, 3)
# [a, b] = array
# a += 1
# print(a)
# print(array) # [1,2,3] if replicated. [0,1,2] if not


# # can i add 2 2d arrays using the addition operator
# # Conclusion: Yes
# a = np.arange(9).reshape(3,3)
# b = np.arange(9).reshape(3,3)
# c = a + b
# print(c)


# # if I use array deconstruction to grab a subarray from an np array, then modify the main array. Does that replicate to the subarray
# # Conclusion: Yes
# array = np.arange(6).reshape(2, 3)
# [a, b] = array
# array[0] += 1
# print(a)
# print(array) # [1,2,3] if replicated. [0,1,2] if not

