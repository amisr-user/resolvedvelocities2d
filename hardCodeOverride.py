import numpy as np

def hardCodeOverride():
    
    numEvents = 1
    numInst = 1
    numMatrices = 9
    numErrors = 0
    numDicts = 9
    
    numProcesses = numEvents * numMatrices * numDicts + numInst * numErrors * numMatrices 
    
    hardCode= [dict() for x in range(numProcesses)]
    
    numReal = numEvents * numMatrices * numDicts
    numSim = numInst * numErrors * numMatrices
    
    simFlag = []
    realFlag = []
    
    for i in range(numReal):
        simFlag.append(0)
    for i in range(numReal, numReal + numSim):
        simFlag.append(1)
        
    realFlag = [not i for i in simFlag]
    
    
    #event1 = "pfisr/20121107.001_lp_5min-cal/20121107.001_lp_5min-cal.h5"
    event1 = "pfisr/20121107.001_lp_1min-cal/20121107.001_lp_1min-cal.h5"
    
   # event2 = "risrn/20090915.001_lp_3min-cal/20090915.001_lp_3min-cal.h5"
#    event3 = "pfisr/20121124.001_lp_1min/20121124.001_lp_1min.h5"
#    event4 = "risrn/20100313.001_lp_3min-cal/20100313.001_lp_3min-cal.h5"
#    event5 = "risrn/20120219.001_lp_2min/20120219.001_lp_2min.h5"
    
    #inst1 = "pfisr/20121107.001_lp_5min-cal/20121107.001_lp_5min-cal.h5"
    inst1 = "pfisr/20121107.001_lp_1min-cal/20121107.001_lp_1min-cal.h5"
    
 #   inst2 = "risrn/20090915.001_lp_3min-cal/20090915.001_lp_3min-cal.h5"
    
    matrix1 = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    matrix2 = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    matrix3 = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    
    
    matrix4 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    matrix5 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    matrix6 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    
    matrix7 = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    matrix8 = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    matrix9 = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    
  #  errVS = [0, 0, 0.1]
  #  errS = [0.0018/50.0, -0.3194/50.0, 34.07/50.0]
  #  errR = [0.00036, -0.3194/5.0, 34.07/5.0]
    
    eventList = [event1] #, event2, event3, event4, event5]
    instList = [inst1] #, inst2]
    instLabelList = ["p_1min"]
    
    matrixList = [matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8, matrix9]
    matrixLabelList = ["110000", "001100", "000011", "000110", "001001", "100100", "011000", "100001", "010010"]
  
    
    eventLabelList = ["p121107_1min"] #, "r090915", "p121124", "r100313", "r120219"]    
    xyx = yxx = xyy = yxy = xy = yx = 0
    
    x, y = (1.0, 1.0)
    xx, yy = (1.0, 1.0)
    xxx, xxy, yyx, yyy = (1.0, 1.0, 1.0, 1.0)
    derivDict0 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivativeWeightDict = [derivDict0, derivDict0, derivDict0, derivDict0, derivDict0, derivDict0, derivDict0, derivDict0, derivDict0]
    
    x, y = (1.0, 0.5)
    xx, yy = (1.0, 0.5)
    xxx, xxy, yyx, yyy = (1.0, 1.0, 0.5, 0.5)
    derivDict1 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivDict1Label = "x1y05_xx1y05_xxx1xxy1yyx05yyy05"
    
    x, y = (0.5, 1.0)
    xx, yy = (0.5, 1.0)
    xxx, xxy, yyx, yyy = (0.5, 0.5, 1.0, 1.0)
    derivDict2 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivDict2Label = "x05y1_xx05y1_xxx05xxy05yyx1yyy1"
    
    x, y = (1.0, 0.75)
    xx, yy = (1.0, 0.75)
    xxx, xxy, yyx, yyy = (1.0, 1.0, 0.75, 0.75)
    derivDict3 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivDict3Label = "x1y75_xx1y75_xxx1xxy1yyx75yyy75"
    
    x, y = (0.75, 1.0)
    xx, yy = (0.75, 1.0)
    xxx, xxy, yyx, yyy = (0.75, 0.75, 1.0, 1.0)
    derivDict4 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivDict4Label = "x75y1_xx75yy1_xxx75xxy75yyx1yyy1"
    
    x, y = (1.0, 1.0)
    xx, yy = (1.25, 0.75)
    xxx, xxy, yyx, yyy = (0.75, 0.75, 1.25, 1.25)
    derivDict5 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivDict5Label = "x1y1_xx125yy75_xxx75xxy75yyx125yyy125"
    
    x, y = (1.0, 1.0)
    xx, yy = (0.75, 1.25)
    xxx, xxy, yyx, yyy = (1.25, 1.25, 0.75, 0.75)
    derivDict6 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivDict6Label = "x1y1_xx75yy125_xxx125xxy125yyx75yyy75"
    
    x, y = (0.0, 1.0)
    xx, yy = (0.0, 1.0)
    xxx, xxy, yyx, yyy = (0.0, 0.0, 1.0, 1.0)
    derivDict7 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivDict7Label = "x0y1_xx0yy1_xxx0xxy0yyx1yyy1"
    
    x, y = (1.0, 0.0)
    xx, yy = (1.0, 0.0)
    xxx, xxy, yyx, yyy = (1.0, 1.0, 0.0, 0.0)
    derivDict8 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    derivDict8Label = "x1y0_xx1yy0_xxx1xxy1yyx0yyy0"
    
    derivDictOptions = [derivDict1, derivDict2, derivDict3, derivDict4, derivDict5, derivDict6, derivDict7, derivDict8]
    derivDictLabel = [derivDict1Label, derivDict2Label, derivDict3Label, derivDict4Label, derivDict5Label, derivDict6Label, derivDict7Label, derivDict8Label]
    
    for i in range(0, len(matrixList)):
        for j in range(0, 8):
            matrixList.append(matrixList[i])
            derivativeWeightDict.append(derivDictOptions[j])
            newLabel = matrixLabelList[i] + "_" + derivDictLabel[j]
            matrixLabelList.append(newLabel)
    

    
    index = 0
    for i in range(numEvents):
        for j in range(numMatrices * numDicts):
            hardCode[index]["pathToData"] = eventList[i]
            hardCode[index]["masksc"] = matrixList[j]
            hardCode[index]["Perr"] = [0, 0, 0]
            hardCode[index]["label"] = eventLabelList[i] + "_" + matrixLabelList[j]
            index += 1
                
    for i in range(numInst):
        for j in range(numErrors):
            for k in range(numMatrices):
                hardCode[index]["pathToData"] = instList[i]
                hardCode[index]["Perr"] = errList[j]
                hardCode[index]["label"] = instLabelList[i] + "_" + errLabelList[j] + "_" + matrixLabelList[k]
                hardCode[index]["masksc"] = matrixList[k]
                index +=1
   

    print derivativeWeightDict
    print matrixLabelList
    print matrixList
    
    
    return numProcesses, simFlag, realFlag, derivativeWeightDict, hardCode
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    x, y = (1.0, 1.0)
    xx, xy, yx, yy = (0.5, 0.5, 1.0, 1.0)
    xxx, xxy, xyx, yxx, xyy, yxy, yyx, yyy = (0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    derivDict2 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    matrixLabelList.append('001001_xx0_xy0_yx1_yy1')



    x, y = (0.0, 0.0)
    xx, xy, yx, yy = (0.75, 0.75, 0.25, 0.25)
    xxx, xxy, xyx, yxx, xyy, yxy, yyx, yyy = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    derivDict3 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #  matrixLabelList.append('001001_xx75_xy75_yx25_yy25')


    x, y = (0.0, 0.0)
    xx, xy, yx, yy = (0.25, 0.25, 0.75, 0.75)
    xxx, xxy, xyx, yxx, xyy, yxy, yyx, yyy = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    derivDict4 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx25_xy25_yx75_yy75')
    # 

    x, y = (0.0, 0.0)
    xx, xy, yx, yy = (1.0, 1.0, 0.1, 0.1)
    xxx, xxy, xyx, yxx, xyy, yxy, yyx, yyy = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    derivDict5 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx1_xy1_yx01_yy01')


    x, y = (0.0, 0.0)
    xx, xy, yx, yy = (0.1, 0.1, 1.0, 1.0)
    xxx, xxy, xyx, yxx, xyy, yxy, yyx, yyy = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    derivDict6 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #  matrixLabelList.append('001001_xx01_xy01_yx1_yy1')




    xyx = 0
    yxx = 0
    xyy = 0
    yxy = 0

    ###
    x, y = (0.0, 0.0)
    xx, xy, yx, yy = (1.0, 1.0, 1.0, 1.0)
    xxx, xxy, yyx, yyy = (1.0, 1.0, 0.0, 0.0)
    derivDict7 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx1_xy1_yx1_yy1_xxx1_xxy1_yyx0_yyy0')

    xxx, xxy, yyx, yyy = (0.0, 0.0, 1.0, 1.0)
    derivDict8 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx1_xy1_yx1_yy1_xxx0_xxy0_yyx1_yyy1')

    xxx, xxy, yyx, yyy = (0.9, 0.9, 0.1, 0.1)
    derivDict9 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #  matrixLabelList.append('001001_xx1_xy1_yx1_yy1_xxx09_xxy09_yyx01_yyy01')

    xxx, xxy, yyx, yyy = (0.1, 0.1, 0.9, 0.9)
    derivDict9 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #  matrixLabelList.append('001001_xx1_xy1_yx1_yy1_xxx01_xxy01_yyx09_yyy09')

    xxx, xxy, yyx, yyy = (0.75, 0.75, 0.25, 0.25)
    derivDict10 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    matrixLabelList.append('001001_xx1_xy1_yx1_yy1_xxx75_xxy75_yyx25_yyy25')

    xxx, xxy, yyx, yyy = (0.25, 0.25, 0.75, 0.75)
    derivDict11 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx1_xy1_yx1_yy1_xxx25_xxy25_yyx75_yyy75')


    xx, xy, yx, yy = (1.0, 1.0, 0.1, 0.1)
    xxx, xxy, yyx, yyy = (1.0, 1.0, 0.0, 0.0)
    derivDict12 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx1_xy1_yx01_yy01_xxx1_xxy1_yyx0_yyy0')

    xxx, xxy, yyx, yyy = (0.0, 0.0, 1.0, 1.0)
    derivDict13 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx1_xy1_yx01_yy01_xxx0_xxy0_yyx1_yyy1')

    xxx, xxy, yyx, yyy = (0.9, 0.9, 0.1, 0.1)
    derivDict14 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #  matrixLabelList.append('001001_xx1_xy1_yx01_yy01_xxx09_xxy09_yyx01_yyy01')

    xxx, xxy, yyx, yyy = (0.1, 0.1, 0.9, 0.9)
    derivDict15 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx1_xy1_yx01_yy01_xxx01_xxy01_yyx09_yyy09')

    xxx, xxy, yyx, yyy = (0.75, 0.75, 0.25, 0.25)
    derivDict16 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #matrixLabelList.append('001001_xx1_xy1_yx01_yy01_xxx75_xxy75_yyx25_yyy25')

    xxx, xxy, yyx, yyy = (0.25, 0.25, 0.75, 0.75)
    derivDict17 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx1_xy1_yx01_yy01_xxx25_xxy25_yyx75_yyy75')




    xx, xy, yx, yy = (0.1, 0.1, 1.0, 1.0)
    xxx, xxy, yyx, yyy = (1.0, 1.0, 0.0, 0.0)
    derivDict18 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #  matrixLabelList.append('001001_xx01_xy01_yx1_yy1_xxx1_xxy1_yyx0_yyy0')

    xxx, xxy, yyx, yyy = (0.0, 0.0, 1.0, 1.0)
    derivDict19 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx01_xy01_yx1_yy1_xxx0_xxy0_yyx1_yyy1')

    xxx, xxy, yyx, yyy = (0.9, 0.9, 0.1, 0.1)
    derivDict20 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #  matrixLabelList.append('001001_xx01_xy01_yx1_yy1_xxx09_xxy09_yyx01_yyy01')

    xxx, xxy, yyx, yyy = (0.1, 0.1, 0.9, 0.9)
    derivDict21 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #  matrixLabelList.append('001001_xx01_xy01_yx1_yy1_xxx01_xxy01_yyx09_yyy09')

    xxx, xxy, yyx, yyy = (0.75, 0.75, 0.25, 0.25)
    derivDict22 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    # matrixLabelList.append('001001_xx01_xy01_yx1_yy1_xxx75_xxy75_yyx25_yyy25')

    xxx, xxy, yyx, yyy = (0.25, 0.25, 0.75, 0.75)
    derivDict23 = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'xyx': xyx, 'yxx': yxx, 'xyy': xyy, 'yxy': yxy, 'yyx': yyx, 'yyy': yyy}
    #matrixLabelList.append('001001_xx01_xy01_yx1_yy1_xxx25_xxy25_yyx75_yyy75')



    #derivativeWeightDict = [derivDict1, derivDict2, derivDict3, derivDict4, derivDict5, derivDict6, derivDict7, derivDict8, derivDict9, derivDict10, derivDict11, derivDict12, derivDict13, derivDict14, derivDict15, derivDict16, derivDict17, derivDict18, derivDict19, derivDict20, derivDict21, derivDict22, derivDict23]
    #for i in range(numReal + numSim):
    #    derivativeWeightDict.append(derivativeWeightDictEntry)
    
    
    
    derivativeWeightDict = [derivDict1, derivDict2, derivDict1, derivDict2]
    #matrixLabelList = ["mag1_1001_equalWeights", "mag1_1001_xDerivs05_yDerivs1", "mag1_0110_equalWeights", "mag1_0110_xDerivs05_yDerivs1"]