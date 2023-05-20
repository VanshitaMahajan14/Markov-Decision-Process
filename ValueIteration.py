import numpy as np

#p_direction is the probability that the agent moves in the mentioned direction
#p_perpendicular is the probabiltiy that the agent moves perpendicular to the mentioned direction
p_direction = 0.7  
p_perpendicular = 0.15
discount = 0.95
wall = (2,1)

reward_grid = np.array([
    [0,1,-1],
    [0,0,0],
    [0,0,0],
    [0,0,0]
])

U1 = [[0 for j in range(len(reward_grid[0]))] for i in range(len(reward_grid))]

U = [[0 for j in range(len(reward_grid[0]))] for i in range(len(reward_grid))]

Policy_Matrix = [[0 for j in range(len(reward_grid[0]))] for i in range(len(reward_grid))]


U[0][1] = 1
U[0][2] = -1
U1[0][1] = 1
U1[0][2] = -1

print(U)
print(Policy_Matrix)


#function that calculates probability that the agent ends up in the square vertically up from the current square

def calc_upval(U,i,j):
   
    #up
    if(i==0):
        up = U[i][j]
    elif((i-1,j) == wall):  #if wall, agent stays at same position
        up = U[i][j]
    else:
        up = U[i - 1][j]
    
    #left
    if(j==0):
        left = U[i][j]
    elif((i,j-1) == wall):
        left = U[i][j]
    else:
        left = U[i][j-1]
        
    #right
    if(j==2):
        right = U[i][j]
    elif((i,j+1) == wall):
        right = U[i][j]
    else:
        right = U[i][j+1]

    return p_direction*up + p_perpendicular*(left + right)
    

#function that calculates probability that the agent ends up in the square horizontally to the right of the current square
    
def calc_rightval(U,i,j):
    
    #up
    if(i==0):
        up = U[i][j]
    elif((i-1,j) == wall):  #if wall, agent stays at same position
        up = U[i][j]
    else:
        up = U[i - 1][j]
        
    #right
    if(j == 2):
        right = U[i][j]
    elif((i,j+1) == wall):
        right = U[i][j]
    else:
        right = U[i][j+1]
        
    #down
    if(i==3):
        down = U[i][j]
    elif((i+1,j) == wall):
        down = U[i][j]
    else:
        down = U[i + 1][j]
        
    return p_direction*right + p_perpendicular*(up + down)

#function that calculates probability that the agent ends up in the square horizontally to the left of the current square


def calc_leftval(U,i,j):
    
    #up
    if(i==0):
        up = U[i][j]
    elif((i-1,j) == wall):  #if wall, agent stays at same position
        up = U[i][j]
    else:
        up = U[i - 1][j]
        
    #down
    if(i==3):
        down = U[i][j]
    elif((i+1,j) == wall):
        down = U[i][j]
    else:
        down = U[i + 1][j]
        
    #left
    if(j == 0):
        left = U[i][j]
    elif((i,j-1) == wall):
        left = U[i][j]
    else:
        left = U[i][j-1]
        
    return p_direction*left + p_perpendicular*(up + down)


#function that calculates probability that the agent ends up in the square vertically down from the current square

        
def calc_downval(U,i,j):
    
    #left
    if(j == 0):
        left = U[i][j]
    elif((i,j-1) == wall):   #if wall, agent stays at same position
        left = U[i][j]
    else:
        left = U[i][j-1]
    
    #right
    if(j == 2):
        right = U[i][j]
    elif((i,j+1) == wall):   
        right = U[i][j]
    else:
        right = U[i][j+1]
        
    #down
    if(i==3):
        down = U[i][j]
    elif((i+1,j) == wall):   
        down = U[i][j]
    else:
        down = U[i + 1][j]
        
    return p_direction*down + p_perpendicular*(left + right)

epsilon = 0.0001
count = 0
step_cost = -0.04

while(True):
    delta = float('-inf')
    for i in reversed(range(4)):
        for j in range(3):
            U[i][j] = U1[i][j]

    #print(U)   
    count = count + 1
    for i in reversed(range(4)):
        for j in range(3): 
            k = 3-i

            if [i,j] == [2,1]:   #wall condition
                continue

            if(reward_grid[i][j]==1):
                U1[i][j]=1
                U[i][j]=1
                continue
            if(reward_grid[i][j]==-1):
                U1[i][j]=-1
                U[i][j]=-1
                continue
            # if(reward_grid[i][j]==0):
                # U1[i][j]=0
                # U[i][j]=0
                #continue

            up = calc_upval(U,i,j)
            down = calc_downval(U,i,j)
            left = calc_leftval(U,i,j)
            right = calc_rightval(U,i,j)
            max_val = max(up,down,left,right)
            # print(max_val)
            # print(up)

            if(max_val == up):
                Policy_Matrix[i][j] = 'up'
            elif(max_val == down):
                Policy_Matrix[i][j] = 'down'
            elif(max_val == left):
                Policy_Matrix[i][j] = 'left'
            elif(max_val == right):
                Policy_Matrix[i][j] = 'right'


            a = discount * max(up,down,left,right)
            val = a + step_cost
            U1[i][j] = val
            delta = max(delta,abs(U[i][j] - U1[i][j])) 

    #print(U1)
    print("Utility Matrix for iteration ", count)
    print()
    for a in range(4):
        for b in range(3):
            x = U1[a][b]
            x = round(x,3)
            print(x,end=" ")
        print()
    print()


    if delta < epsilon:
        break

print("Policy Matrix")
print(Policy_Matrix)
print()
print("Number of iterations: ")
print(count)       
