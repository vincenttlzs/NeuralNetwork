def findKthLargest(nums, k):
    index = 0
    pivot = nums[index]
    small=[i for i in (nums[:index]+nums[index+1:])if i < pivot]
    large=[i for i in (nums[:index]+nums[index+1:])if i >= pivot]
    if k-1 == len(large):
        return pivot
    elif k-1<len(large):
        return findKthLargest(large,k)
    if k-1 > len(large):
            return findKthLargest(small,k-len(large)-1)


model = load_model('model.h5')   
c = input('Please input the first char(lowercase a~z) of a name: \n')                   
inpu = np.zeros([1,1,27])
inpu[0,0,ord(c)-97] = 1
names = []  
print('Waiting........')           
while len(names) < 20:
    name = [c]
    for n in range(10):               
        infer = model.predict(inpu)
        infern = infer[0,-1].tolist()
        # set a random algorthm. Choose top four elements randomly
        nexchr = chr(infern.index(findKthLargest(infern,np.random.randint(1,5))) + 97) #randnames[k][m]
        # '{' is the EON in here
        if (nexchr == '{'):
            break
        name.append(nexchr)
        inpu3 = np.zeros([1,1,27])
        inpu3[0,0,ord(nexchr)-97] = 1
        inpu = np.hstack((inpu,inpu3))
    name = ''.join(name)
    # remove the name whose size less than 3
    if len(name) <= 2:
        pass                
    elif name in names:
        pass
    else:
        names.append(name)                 
     
print(names)