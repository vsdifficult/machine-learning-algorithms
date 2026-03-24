
def binary_search(arr, l, r, x): 
    if l <= r: 
        mid = int(l + (r-l) / 2)

        if arr[mid] == x: 
            return mid 
        elif arr[mid] > x: 
            return binary_search(arr, l, mid-1, x) 
        else: 
            return binary_search(arr, mid+1, r, x) 
        
    else: 
        return -1 
    
if __name__ == "__main__": 
    x = 5 
    arr = sorted([6, 1, 2, 9, 6, 5, 8, 10, 123, 342, 1234124, 1415124]) 

    print(arr) 
    print("Binary Search") 

    result = binary_search(arr, 0, len(arr)-1, x) 

    if result != -1: 
        print("element {} is found at index {}.".format(x, result)) 
    else: 
        print("element is not found")