def cost_function(x,y,w,b):
    m = len(x)
    cost_sum = 0
    for i in range(m):
        f = (w*x[i]) + b
        cost = (f-y[i])**2
        cost_sum += cost

    return (1/(2*m)) * cost_sum

def gradiant_function(x,y,w,b):
    m = len(x)
    dc_dw = 0
    dc_db = 0

    for i in range(m):
        f = (w*x[i]) + b

        dc_dw += (f - y[i]) * x[i]
        dc_db += (f - y[i])

    dc_dw = (1/m) * dc_dw
    dc_dw = (1/m) * dc_db

    return dc_dw, dc_db


def gradiant_decent(x,y,alpha,iterations):
    w = 0
    b = 0

    for i in range(iterations):
        dc_dw, dc_db = gradiant_function(x,y,w,b)
        
        w = w - (alpha * dc_dw)
        b = b - (alpha * dc_db)

        print(f"Iteration {i}: Cost {cost_function(x,y,w,b)}")

    return w,b




