import numpy as np
import os.path
import re
import ConfigParser
import h5py

def setUp():
    simFlag = []
    realFlag = []
    PerrList = []
    labelList = []
    maskscList = []
    derivWeightDictList = []
    pathToDataList = []
    
    while 1:
        numProcesses = raw_input("Enter the number of processes you'd like to run: ")
        try:
            numProcesses = int(numProcesses)
            break
        except:
            print "Error: number of processes must be an integer."
        
    for i in range(numProcesses):
        print "~~~~~~~~~~~~~~~~~~~~~~ Process #" + str(i + 1) + ": ~~~~~~~~~~~~~~~~~~~~~~"
        
        while 1:
            simOrReal = raw_input("Simulation (S) or real data (R)? ")
            if simOrReal == "S" or simOrReal == "s":
                simFlag.append(1)
                realFlag.append(0)                
                break
        
            elif simOrReal == "R" or simOrReal == "r":
                simFlag.append(0)
                realFlag.append(1)
                break
        
            else:
                print "Error. Invalid input. Please enter S for simulated data or R for real data."
 
        
        
        if simFlag[i]:
            
            while 1:
                
                pathToData = raw_input("Which instrument would you like to run the simulation for? Enter pfisr1 or pfisr5 for 1 minute or 5 minute records, respectively: ")
                
                if pathToData == 'pfisr5':
                    pathToData = "data sources (radar and optical)/pfisr radar data/20121107.001_lp_5min-cal/20121107.001_lp_5min-cal.h5"
                    break
                elif pathToData == 'pfisr1':
                    pathToData = "data sources (radar and optical)/pfisr radar data/20121107.001_lp_1min-cal/20121107.001_lp_1min-cal.h5"
                    break
                else:
                    print "Error: invalid instrument."
            
            while 1: 
                errorStyle =  raw_input("Please enter your error style: ")
                #set error style
         
                if errorStyle.strip()[0] == '[' and errorStyle.strip()[-1] == ']' and len([a.strip() for a in errorStyle.strip()[1:-1].split(",")]) == 3:
                    try:
                        Perr = [float(a.strip()) for a in errorStyle.strip()[1:-1].split(",")]
                        label = "custom"
                        print "Custom error style set to " + str(Perr)
                        break
                    except:
                        print "Incorrect number formatting. Please choose from the following list of preset options or enter a custom 1x3 array of numbers in valid Python list syntax (i.e. [0.45, 0.12, 0.004]). \n"
                        print "errors_vsmall_001 \n errors_small_001 \n errors_small_002 \n errors_small_003 \n errors_small_004 \n errors_reg_001 \n"
       
    
                elif errorStyle == 'errors_vsmall_001' or errorStyle == 'vs':
                    Perr = [0, 0, 0.1];
                    label = "vsmall1"
                    break
        
                elif errorStyle == 'errors_small_001' or errorStyle == 's1':
                    Perr = [0.0018/50.0, -0.3194/50.0, 34.07/50.0];
                    label = "small1"
                    break
        
                elif errorStyle == 'errors_small_002' or errorStyle == 's2':
                    Perr = [0.0018/25.0, -0.3194/25.0, 34.07/25.0];
                    label = "small2"
                    break
        
                elif errorStyle == 'errors_small_003' or errorStyle == 's3':
                    Perr = [0.0018/20.0, -0.3194/20.0, 34.07/20.0];
                    label = "small3"
                    break
        
                elif errorStyle == 'errors_small_004' or errorStyle == 's4':
                    Perr = [0.0018/15.0, -0.3194/15.0, 34.07/15.0];
                    label = "small4"
                    break
        
                elif errorStyle == 'errors_reg_001' or errorStyle == 'r':
                    Perr = [0.00036, -0.3194/5.0, 34.07/5.0];
                    label = "reg1"
                    break
        
                else:
                     print "Error: invalid error style name. Please choose from the following list of preset options or enter a custom 1x3 array in valid Python list syntax (i.e. [0.45, 0.12, 0.004]). \n"
                     print "errors_vsmall_001 \n errors_small_001 \n errors_small_002 \n errors_small_003 \n errors_small_004 \n errors_reg_001 \n"
        
        else: #if real data
            Perr = [0, 0, 0]
            
            while 1:
                
                pathToDataUser = raw_input("Please provide the path to the real data source file or enter one of the following options: \n'pfisr121107_1'\n'pfisr121107_5'\n'pfisr121124'\n'pfisr151012'\n")
                
                if pathToDataUser == 'pfisr121107_5' or pathToDataUser == 'p5':
                    pathToDataUser = 'p5'
                    pathToData = "data sources (radar and optical)/pfisr radar data/20121107.001_lp_5min-cal/20121107.001_lp_5min-cal.h5"
                    break
                elif pathToDataUser == 'pfisr121107_1':
                    pathToDataUser = 'p1'
                    pathToData = "data sources (radar and optical)/pfisr radar data/20121107.001_lp_1min-cal/20121107.001_lp_1min-cal.h5"
                    break
                elif pathToDataUser == 'pfisr121124':
                    pathToData = "data sources (radar and optical)/pfisr radar data/20121124.001_lp_1min/20121124.001_lp_1min.h5"
                    break
                elif pathToDataUser == 'pfisr151012':
                    pathToData =  "data sources (radar and optical)/pfisr radar data/20151012.001_lp_3min-cal/20151012.001_lp_3min-cal.h5"
                    break
                elif os.path.isfile(pathToData):
                    break
                else:
                    print "Error: file does not exist. Please enter a valid path."
                    
            label = pathToDataUser
        
        PerrList.append(Perr)
        pathToDataList.append(pathToData)
        
        
        #set masking style using user input.
        #three accepted format styles:
            # 110011
            # 1.0 1.0 0.0 0.0 1.0 1.0
            # [[1.0, 1.0], [0.0, 0.0] ,[1.0, 1.0]]
        DONE = 0
        while not DONE:
        
            derivFlag = raw_input("Would you like to weight derivatives in diff. directions differently in the G matrix mask? Y/N: ")
            if derivFlag == "Y" or derivFlag == "y":
                derivativeWeightDict = {}
                print "Enter derivative weights:"
                for derivWeight in ['x', 'y', 'xx', 'xy', 'yx', 'yy', 'xxx', 'xxy', 'yyx', 'yyy']:
                    derivativeWeightDict[derivWeight] = setValue(derivWeight)
                masksc = np.array([[1, 1], [1, 1], [1, 1]])
                label += "_x" + str(derivativeWeightDict['x']) + "y" + str(derivativeWeightDict['y']) + "xx" + str(derivativeWeightDict['xx']) + "xy" + str(derivativeWeightDict['xy']) + "yx" + str(derivativeWeightDict['yx']) + "yy" + str(derivativeWeightDict['yy']) + "xxx" + str(derivativeWeightDict['xxx']) + "xxy" + str(derivativeWeightDict['xxy']) + "yyx" + str(derivativeWeightDict['yyx']) + "yyy" + str(derivativeWeightDict['yyy'])

                DONE = 1
                break
                   
        
            elif derivFlag == "N" or derivFlag == "n":
                while 1: 
                    maskStyle = raw_input("Please enter your desired G matrix masking style (e.g. [[0, 0], [1, 0], [0, 1]] or 001001): ")
                    masksc = ""
        
                    #check 110011 style
                    try:
                        masksc = map(float, list(maskStyle.replace(" ", "")))
                        if len(masksc) == 6:
                            masksc = np.reshape(np.asarray(masksc), (3,2))
                            label += "_" + str(maskStyle)
                            DONE = 1
                            break
                    except:
                        pass
                    # check 1.0 1.0 0.0 0.0 1.0 1.0 style 
                    try:
                        masksc = map(float, maskStyle.split())
                        if len(masksc) == 6:
                            masksc = np.reshape(np.asarray(masksc), (3,2))
                            DONE = 1
                            label += "_" + str(maskStyle)
                            break
                    except:
                        pass
    
                    # check [[1.0, 1.0], [0.0, 0.0] ,[1.0, 1.0]] style
                    try:
                        masksc = [[float(d) for d in x.split(",")] for x in re.compile("\].*?\[").split(maskStyle.strip().strip("[").strip("]"))]
                    except:
                            print "Input Format Error" 
        
                    masksc = np.asarray(masksc)
    
                    if masksc.shape == (3, 2):
                        DONE = 1
                        label += "_" + str(maskStyle)
                        break
            
                    print "Shape error: mask must be 3x2 array."
                    
                #default dict when mask is given is all ones, because mask will multiply 
                x, y = (1.0, 1.0)
                xx, xy, yx, yy = (1.0, 1.0, 1.0, 1.0)
                xxx, xxy, yyx, yyy = (1.0, 1.0, 1.0, 1.0)
                derivativeWeightDict = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'yyx': yyx, 'yyy': yyy}
                    
            else:
                print "Error: please enter either Y/y or N/n."

        
           
        maskscList.append(masksc)
        derivWeightDictList.append(derivativeWeightDict)
        labelList.append(label)
        
    
    return numProcesses, simFlag, realFlag, PerrList, labelList, maskscList, derivWeightDictList, pathToDataList
    
#RHV new ini setUp:
def setUpini(inifilename,filepath):
    config=ConfigParser.ConfigParser()
    config.read(inifilename)

    simFlag = []
    realFlag = []
    PerrList = []
    labelList = []
    maskscList = []
    derivWeightDictList = []
    pathToDataList = []
    
    numProcesses=1
        
    for i in range(numProcesses):
        print "~~~~~~~~~~~~~~~~~~~~~~ Process #" + str(i + 1) + ": ~~~~~~~~~~~~~~~~~~~~~~"
        

        simOrReal = "R"
        if simOrReal == "S" or simOrReal == "s":
            simFlag.append(1)
            realFlag.append(0)                
    
        elif simOrReal == "R" or simOrReal == "r":
            simFlag.append(0)
            realFlag.append(1)

 
        Perr = [0, 0, 0]
        
        pathToData = config.get('io','dataPath')+filepath
            
        label = re.split('/|\.h5',pathToData)[-2]+'_2dVEF'
            
        PerrList.append(Perr)
        pathToDataList.append(pathToData)
        
        
        #set masking style using user input.
        #three accepted format styles:
            # 110011
            # 1.0 1.0 0.0 0.0 1.0 1.0
            # [[1.0, 1.0], [0.0, 0.0] ,[1.0, 1.0]]
        DONE = 0
        while not DONE:
        
            derivFlag = config.get('deriv','derivFlag')        
            if derivFlag == "Y" or derivFlag == "y":
                derivativeWeightDict = {}
                print "Enter derivative weights:"
                for derivWeight in ['x', 'y', 'xx', 'xy', 'yx', 'yy', 'xxx', 'xxy', 'yyx', 'yyy']:
                    derivativeWeightDict[derivWeight] = config.getfloat('deriv',derivWeight)
                masksc = np.array([[1, 1], [1, 1], [1, 1]])
                label += "_x" + str(derivativeWeightDict['x']) + "y" + str(derivativeWeightDict['y']) + "xx" + str(derivativeWeightDict['xx']) + "xy" + str(derivativeWeightDict['xy']) + "yx" + str(derivativeWeightDict['yx']) + "yy" + str(derivativeWeightDict['yy']) + "xxx" + str(derivativeWeightDict['xxx']) + "xxy" + str(derivativeWeightDict['xxy']) + "yyx" + str(derivativeWeightDict['yyx']) + "yyy" + str(derivativeWeightDict['yyy'])

                DONE = 1
                break
                   
        
            elif derivFlag == "N" or derivFlag == "n":
                while 1: 
                    maskStyle = config.get('deriv','maskStyle')
                    masksc = ""
        
                    #check 110011 style
                    try:
                        masksc = map(float, list(maskStyle.replace(" ", "")))
                        if len(masksc) == 6:
                            masksc = np.reshape(np.asarray(masksc), (3,2))
                            label += "_" + str(maskStyle)
                            DONE = 1
                            break
                    except:
                        pass
                    # check 1.0 1.0 0.0 0.0 1.0 1.0 style 
                    try:
                        masksc = map(float, maskStyle.split())
                        if len(masksc) == 6:
                            masksc = np.reshape(np.asarray(masksc), (3,2))
                            DONE = 1
                            label += "_" + str(maskStyle)
                            break
                    except:
                        pass
    
                    # check [[1.0, 1.0], [0.0, 0.0] ,[1.0, 1.0]] style
                    try:
                        masksc = [[float(d) for d in x.split(",")] for x in re.compile("\].*?\[").split(maskStyle.strip().strip("[").strip("]"))]
                    except:
                            print "Input Format Error" 
        
                    masksc = np.asarray(masksc)
    
                    if masksc.shape == (3, 2):
                        DONE = 1
                        label += "_" + str(maskStyle)
                        break
            
                    print "Shape error: mask must be 3x2 array."
                    
                #default dict when mask is given is all ones, because mask will multiply 
                x, y = (1.0, 1.0)
                xx, xy, yx, yy = (1.0, 1.0, 1.0, 1.0)
                xxx, xxy, yyx, yyy = (1.0, 1.0, 1.0, 1.0)
                derivativeWeightDict = {'x': x, 'y': y, 'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy, 'xxx': xxx, 'xxy': xxy, 'yyx': yyx, 'yyy': yyy}
                    
            else:
                print "Error: please enter either Y/y or N/n."

        
           
        maskscList.append(masksc)
        derivWeightDictList.append(derivativeWeightDict)
        labelList.append(label)
        
        
    opts={}
    opts['numProcesses']=numProcesses
    opts['simFlag']=simFlag
    opts['realFlag']=realFlag
    opts['Perr']=PerrList
    opts['label']=labelList
    opts['masksc']=maskscList
    opts['derivativeWeightDict']=derivWeightDictList
    opts['pathsToData']=pathToDataList
    
    try:
        opts['recStart']=config.getint('recs','recStart')
        if opts['recStart']<0:
            raise ValueError
    except (ValueError,ConfigParser.NoOptionError):  
        opts['recStart']=0
        
    try:        
        opts['recStop']=config.getint('recs','recStop')
        if opts['recStop']<0:
            raise ValueError
    except (ValueError,ConfigParser.NoOptionError):
        #read the data to find how many records there are
        print 'opening file', pathToDataList[0]
        dat = h5py.File(pathToDataList[0])
        opts['recStop']=np.asarray(dat['Time/UnixTime']).shape[0]        
    
    #RHV useIokIntersection was previously hard-coded as 1. I do not recommend that
    opts['useIokIntersection']=config.getint('Iok','useIokIntersection')

    return opts
    
def setValue(variableNameString):
    while 1:
        num = raw_input(variableNameString + " = ")
        try: return float(num); break
        except: print "Error. Invalid input."











































    

      
