--READ THE README.md FILE FOR DOCUMENTATION AND OTHER STUFF!!!!

_G["lnn"] = {}
_G["lnn"]["debug"] = {}
_G["lnn"]["data"] = {}

function lnn.asserttype(variable,variablename,thetype)
    --check for errors in the function that checks for errors.
    if type(thetype)  ~= "string" or type(variablename) ~= "string" then
        error("variablename and thetype must be a string.")
    end

    --give an error if false or nil
    if type(variable) ~= thetype or variable == nil then
        error(string.format("%s (%s) is not a %s or is nil. Type: %s", variablename, tostring(variable), thetype, type(variable)))
    end
end

function lnn.assertsize(a,b,aname,bname)
    --check for errors in the function that checks for errors but different.
    if type(a) ~= "table" or type(b) ~= "table" then
        error("a and b must be a table.")
    end
    if type(aname) ~= "string" or type(bname) ~= "string" then
        error("aname and bname must be a string.")
    end

    --give an error they're not the same size.
    if #a ~= #b then
        error(aname.." ("..#a..") is not the same size as "..bname.." ("..#b..").")
    end
end

function lnn.findintable(item,table)
    --check for errors
    lnn.asserttype(table,"table","table")

    --do the stuff
    for i = 1,#table do
        if table[i] == item then
            return true
        end
    end
    return false
end

function lnn.sigmoid(x,derivative)
    --check for errors
    lnn.asserttype(x,"x","number")
    lnn.asserttype(derivative,"derivative","boolean")

    --do the stuff
    if derivative then
        return (1 / (1 + math.exp(-x))) * (1-(1 / (1 + math.exp(-x))))
    else
        return 1 / (1 + math.exp(-x))
    end
end

function lnn.tanh(x,derivative)
    --check for errors
    lnn.asserttype(x,"x","number")
    lnn.asserttype(derivative,"derivative","boolean")

    --do the stuff
    if derivative then
        return 1 - (((math.exp(2*x) - 1)/(math.exp(2*x) + 1))^2)
    else
        return (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    end
end

function lnn.relu(x,derivative)
    --check for errors
    lnn.asserttype(x,"x","number")
    lnn.asserttype(derivative,"derivative","boolean")

    --do the stuff
    if derivative then
        if x < 0 then
            return 0
        else
            return 1
        end
    else
        return math.max(0,x)
    end
end

function lnn.leakyrelu(x,derivative)
    --check for errors
    lnn.asserttype(x,"x","number")
    lnn.asserttype(derivative,"derivative","boolean")

    --do the stuff
    if derivative then
        if x > 0 then
            return 1
        else
            return 0.01
        end
    else
        if x > 0 then
            return x
        else
            return x*0.01
        end
    end
end

function lnn.elu(x,derivative,alpha)
    --check for errors
    lnn.asserttype(x,"x","number")
    lnn.asserttype(derivative,"derivative","boolean")

    --do the stuff
    if derivative then
        if x < 0 then
            return alpha*math.exp(x)
        else
            return 1
        end
    else
        if x < 0 then
            return math.exp(x)-1
        else
            return x
        end
    end
end

function lnn.swish(x,derivative,beta)
    --check for errors
    lnn.asserttype(x,"x","number")
    lnn.asserttype(derivative,"derivative","boolean")

    --do the stuff
    if derivative then
        return x/(1+math.exp(-beta*x))+lnn.sigmoid(x,false)*(1-x/(1+math.exp(-beta*x)))
    else
        return x/(1+math.exp(-beta*x))
    end
end

function lnn.initialize(id,activation,insize,layercount,outcount)
    --check for errors
    lnn.asserttype(id,"id","string")
    lnn.asserttype(activation,"activation","string")
    lnn.asserttype(insize,"insize","number")
    lnn.asserttype(layercount,"layercount","number")
    lnn.asserttype(outcount,"outcount","number")

    --check if the id already exists
    if _G[id] ~= nil then
        error("id ("..id..") already exists.")
    end

    --available activation functions
    local activationtable = {"sig","tanh","relu","lrelu","elu","swish"}
    
    --check if the activation is a valid activation function
    if not lnn.findintable(activation,activationtable) then
        error(activation.." is not a valid activation function, available activation functions are: 'sig', 'tanh', 'relu', 'lrelu', 'elu' and 'swish'.")
    end

    --initialize the neural network

    --initialize the neural network data
    _G[id] = {}
    _G[id]["activation"] = activation
    _G[id]["layercount"] = layercount
    _G[id]["outcount"] = outcount
    _G[id]["insize"] = insize
    _G[id]["gradient"] = {}
    _G[id]["gradient"]["gradw"] = {}
    _G[id]["gradient"]["gradb"] = {}
    _G[id]["id"] = id
    _G[id]["weight"] = {}
    _G[id]["bias"] = {}
    _G[id]["current"] = {}

    --initialize the neural network layers

    local amounttofill = 0
    
    --check if layercount is 0
    if layercount == 0 then
        --create the tables for the output weights
        _G[id]["weight"]["ow"] = {}
        for i = 1,insize*outcount do
            _G[id]["weight"]["ow"][i] = math.random(0,100)/100
        end

        --create the values for the output node values (bias and current)
        _G[id]["bias"]["ob"] = {}
        _G[id]["current"]["o"] = {}
        for i = 1,outcount do
            _G[id]["bias"]["ob"][i] = math.random(0,100)/100
            _G[id]["current"]["o"][i] = 0
        end
        
        return
    end

    --create the tables for the node values (bias and current)
    for i = 1,layercount do
        local ctablename = "c"..i
        local btablename = "b"..i
        _G[id]["current"][ctablename] = {}
        _G[id]["bias"][btablename] = {}
        
        amounttofill = math.ceil(((outcount - insize) * i / (layercount - 0)) + insize)

        for a = 1,amounttofill do
            _G[id]["current"][a] = 0.0
            _G[id]["bias"][a] = math.random(0,100)/100
        end
    end

    --create the tables for the connection values (weight)
    for i = 1,layercount + 1 do
        local wtablename = "w"..i
        _G[id]["weight"][wtablename] = {}
        
        if i > 1 then --get the amount to fill
            amounttofill = math.ceil(((outcount - insize) * (i - 1) / (layercount - 0)) + insize)*math.ceil(((outcount - insize) * i / (layercount - 0)) + insize)
        else
            amounttofill = insize*math.ceil(((outcount - insize) * i / (layercount - 0)) + insize)
        end

        for a = 1,amounttofill do
            _G[id]["weight"][wtablename][a] = math.random(0,100)/100
        end
    end
    
    --create the tables for the output (bias and current)
    _G[id]["bias"]["ob"] = {}
    _G[id]["current"]["o"] = {}
    for i = 1,outcount do
        _G[id]["bias"]["ob"][i] = math.random(0,100)/100
        _G[id]["current"]["o"][i] = 0.0
    end

    --create the tables for the output connection (weight)
    _G[id]["weight"]["ow"] = {}
    for i = 1,outcount*math.ceil(((outcount - insize) * layercount / (layercount - 0)) + insize) do
        _G[id]["weight"]["ow"][i] = math.random(0,100)/100
    end
end

function lnn.forwardpass(id,intable)
    --check if the id doesn't exist
    if _G[id] == nil then
        error("id ("..id..") doesnt exist.")
    end
    
    --check for errors
    if #intable ~= _G[id]["insize"] then
        error("intable ("..#intable..") is not the same size as the intable when id ("..id..") was initialized (".._G[id]["insize"]").")
    end
    lnn.asserttype(id,"id","string")
    lnn.asserttype(intable,"intable","table")
    
    --declare the functions
    local function getlayer(lastlayer,nextlayer,weights,biases)
        --declare the variables
        local sum = 0

        --get the sum of the connected weights to the current node we are on and replace the nextlayer
        if _G[id]["activation"] == "sig" then
            for a = 1,#nextlayer do
                for i = 1,#lastlayer do
                    sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
                end
                nextlayer[a] = lnn.sigmoid(-sum+biases[a],false)
                sum = 0
            end
        elseif _G[id]["activation"] == "tanh" then
            for a = 1,#nextlayer do
                for i = 1,#lastlayer do
                    sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
                end
                nextlayer[a] = lnn.tanh(-sum+biases[a],false)
                sum = 0
            end
        elseif _G[id]["activation"] == "relu" then
            for a = 1,#nextlayer do
                for i = 1,#lastlayer do
                    sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
                end
                nextlayer[a] = lnn.relu(-sum+biases[a],false)
                sum = 0
            end
        elseif _G[id]["activation"] == "lrelu" then
            for a = 1,#nextlayer do
                for i = 1,#lastlayer do
                    sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
                end
                nextlayer[a] = lnn.leakyrelu(-sum+biases[a],false)
                sum = 0
            end
        elseif _G[id]["activation"] == "elu" then
            for a = 1,#nextlayer do
                for i = 1,#lastlayer do
                    sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
                end
                nextlayer[a] = lnn.elu(-sum+biases[a],false,1)
                sum = 0
            end
        elseif _G[id]["activation"] == "swish" then
            for a = 1,#nextlayer do
                for i = 1,#lastlayer do
                    sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
                end
                nextlayer[a] = lnn.swish(-sum+biases[a],false,0.8)
                sum = 0
            end
        else
            error(string.format("id %s has an invalid activation function? (%s)",id,_G[id]["activation"]))
        end
    end

    --do the stuff

    --check if layercount is 0
    if _G[id]["layercount"] == 0 then
        getlayer(intable,_G[id]["current"]["o"],_G[id]["weight"]["ow"],_G[id]["bias"]["ob"])
        return _G[id]["current"]["o"]
    end
    
    --if there's hidden layers
    getlayer(intable,_G[id]["current"]["c1"],_G[id]["weight"]["w1"],_G[id]["bias"]["b1"]) --input layer to first hidden
    for i = 2,_G[id]["layercount"] do --rest of the hidden layers
        getlayer(_G[id]["current"]["c"..i-1],_G[id]["current"]["c"..i],_G[id]["weight"]["w"..i],_G[id]["bias"]["b"..i])
    end
    getlayer(_G[id]["current"]["c".._G[id]["layercount"]],_G[id]["current"]["o"],_G[id]["weight"]["ow"],_G[id]["bias"]["ob"])

    return _G[id]["current"]["o"]
end

function lnn.adjust(id,intable,output,expectedoutput,learningrate)
    --check for errors
    lnn.asserttype(id,"id","string")
    lnn.asserttype(intable,"intable","table")
    lnn.asserttype(output,"output","table")
    lnn.asserttype(expectedoutput,"expectedoutput","table")
    lnn.asserttype(learningrate,"learningrate","number")
    
    lnn.assertsize(output,expectedoutput,"out","expectedout")
    if _G[id] == nil then
        error("id ("..id..") doesn't exist.")
    end
    if _G[id]["insize"] ~= #intable then
        error("insize (".._G[id]["insize"]..") is not the same size as intable ("..#intable..").")
    end

    --declare the variables
    local gradw = {}
    local gradb = {}
    local gradwsum = 0
    local gradbsum = 0
    local weightedsum = 0
    local da_wsum = 0

    --get the sum of the weighted inputs
    for a = 1,#_G[id]["current"]["c1"] do
        for i = 1,#intable do
            weightedsum = weightedsum + _G[id]["weight"]["w1"][i+((a-1)*#intable)]*intable[i]
        end
    end

    --get gradw
    for i = 1,#output do
        gradw[i] = ((output[i]-expectedoutput[i])^2*da_wsum)*learningrate+((output[i]-expectedoutput[i])*learningrate)
    end

    --get gradb
    for i = 1,#output do
        gradb[i] = ((expectedoutput[i]-output[i])*da_wsum)*learningrate+((expectedoutput[i]-output[i])*learningrate)
    end

    --get gradwsum
    for i = 1,#output do
        gradwsum = gradwsum + gradw[i]
    end

    --get gradbsum
    for i = 1,#output do
        gradbsum = gradbsum + gradb[i]
    end

    --update the data on _G[id]
    _G[id]["gradient"]["gradw"] = gradw
    _G[id]["gradient"]["gradb"] = gradb

    --get da_wsum
    if _G[id]["activation"] == "sig" then
        da_wsum = lnn.sigmoid(weightedsum,true)
    elseif _G[id]["activation"] == "tanh" then
        da_wsum = lnn.tanh(weightedsum,true)
    elseif _G[id]["activation"] == "relu" then
        da_wsum = lnn.relu(weightedsum,true)
    elseif _G[id]["activation"] == "lrelu" then
        da_wsum = lnn.leakyrelu(weightedsum,true)
    elseif _G[id]["activation"] == "elu" then
        da_wsum = lnn.elu(weightedsum,true,1)
    elseif _G[id]["activation"] == "swish" then
        da_wsum = lnn.swish(weightedsum,true,0.8)
    else
        error(string.format("id %s has an invalid activation function? (%s)",id,_G[id]["activation"]))
    end
    
    --adjust weights

    --adjust the output layer weights
    for a = 1,#output do
        for i = 1,#_G[id]["current"]["c".._G[id]["layercount"]] do
            _G[id]["weight"]["ow"][i+((a-1)*#_G[id]["current"]["c".._G[id]["layercount"]])] = _G[id]["weight"]["ow"][i+((a-1)*#_G[id]["current"]["c".._G[id]["layercount"]])] - gradw[i]
        end
    end

    --adjust the rest of the weights
    for b = _G[id]["layercount"],1,-1 do
        for i = 1,#_G[id]["weight"]["w"..b] do
            _G[id]["weight"]["w"..b][i] = _G[id]["weight"]["w"..b][i] - gradwsum
        end
    end

    --adjust biases

    --adjust the output layer biases
    for i = 1,#output do
        _G[id]["bias"]["ob"][i] = _G[id]["bias"]["ob"][i] - gradb[i]
    end

    --adjust the rest of the biases
    for b = _G[id]["layercount"],1,-1 do
        for i = 1,#_G[id]["bias"]["b"..b] do
            _G[id]["bias"]["b"..b][i] = _G[id]["bias"]["b"..b][i] - gradbsum
        end
    end
end

function lnn.getmse(output,expectedoutput)
    --check for errors
    lnn.asserttype(output,"output","table")
    lnn.asserttype(expectedoutput,"expectedutput","table")

    lnn.assertsize(output,expectedoutput,"output","expectedoutput")

    --declare the variables
    local mse = 0

    --do the stuff
    for i = 1,#output do
         mse = mse + (expectedoutput[i] - output[i])^2
    end
    return mse/#output
end

function lnn.getsse(output,expectedoutput)
    --check for errors
    lnn.asserttype(output,"output","table")
    lnn.asserttype(expectedoutput,"expectedutput","table")

    lnn.assertsize(#output,#expectedoutput,"output","expectedoutput")

    --declare the variables
    local sse = 0

    --do the stuff
    for i = 1,#output do
        sse = (output[i]-expectedoutput[i])^2
    end
    return sse/#output
end

function lnn.getrmse(output,expectedoutput)
    --check for errors
    lnn.asserttype(output,"output","table")
    lnn.asserttype(expectedoutput,"expectedutput","table")

    lnn.assertsize(output,expectedoutput,"output","expectedoutput")

    --declare the variables
    local rmse = 0

    --do the stuff
    for i = 1,#output do
        rmse = rmse + (expectedoutput[i] - output[i])^2
    end

    return math.sqrt((rmse/#output))
end

function lnn.getcrossentropy(output,expectedoutput)
    --check for errors.
    lnn.asserttype(output,"output","table")
    lnn.asserttype(expectedoutput,"expectedoutput","table")

    lnn.assertsize(output,expectedoutput,"output","expectedoutput")

    --declare the variables
    local sum = 0

    --do the stuff
    for i = 1,#output do
        if expectedoutput[i]+0.01 or output[i]+0.01 <= 0 then print("WARNING: All values put into binary cross entropy function must be bigger than 0 otherwise it will return 'nan'.") end
        sum = sum + (expectedoutput[i]+0.01*math.log(output[i]+0.01)) + (1-expectedoutput[i]+0.01) * math.log(1-output[i]+0.01)
    end
    return -sum
end

function lnn.getbinarycrossentropy(output,expectedoutput)
    --check for errors.
    lnn.asserttype(output,"output","table")
    lnn.asserttype(expectedoutput,"expectedoutput","table")

    lnn.assertsize(output,expectedoutput,"output","expectedoutput")

    --declare the variables
    local sum = 0

    --do the stuff
    for i = 1,#output do
        if output[i]+0.01 or expectedoutput[i]+0.01 <= 0 then print("WARNING: All values put into binary cross entropy function must be bigger than 0 or otherwise it will return 'nan'.") end
        sum = sum + (output[i]*math.log(expectedoutput[i]+0.01)) + ((1-output[i]+0.01)*math.log(1-expectedoutput[i]+0.01))
    end
    
    return sum/-#output
end

function lnn.getcategoricalcrossentropy(output,expectedoutput)
    --check for errors.
    lnn.asserttype(output,"output","table")
    lnn.asserttype(expectedoutput,"expectedoutput","table")

    lnn.assertsize(output,expectedoutput,"output","expectedoutput")

    --declare the variables
    local sum = 0

    --do the stuff
    for i = 1,#output do
        if output[i]+0.01 or expectedoutput[i]+0.01 <= 0 then print("WARNING: All values put into binary cross entropy function must be bigger than 0 or otherwise it will return 'nan'.") end
        sum = sum + output[i]+0.01 * math.log(expectedoutput[i]+0.01)
    end

    return -sum
end

function lnn.gethingeloss(output,expectedoutput)
    --check for errors.
    lnn.asserttype(output,"output","table")
    lnn.asserttype(expectedoutput,"expectedoutput","table")
    
    lnn.assertsize(output,expectedoutput,"output","expectedoutput")

    --declare the variables
    local sum = 0

    --do the stuff
    for i = 1,#output do
        sum = sum + math.max(0,1-output[i]*expectedoutput[i])
    end

    return sum/#output
end

--either debugging or visualizing, could be used for both.

function lnn.debug.returnweights(id)
    --check for errors.
    lnn.asserttype(id,"id","string")

    if _G[id] == nil then
        error("id ("..id..") doesn't exist.")
    end

    --declare the variables
    local returntable = {}

    --do the stuff
    for i = 1,_G[id]["layercount"] do
        returntable[i] = table.pack(_G[id]["bias"]["w"..i])
    end
    returntable[#returntable+1] = table.pack(_G[id]["bias"]["ow"])

    return returntable
end

function lnn.debug.returnbiases(id)
    --check for errors.
    lnn.asserttype(id,"id","string")

    if _G[id] == nil then
        error("id ("..id..") doesn't exist.")
    end

    --declare the variables
    local returntable = {}

    --do the stuff
    for i = 1,_G[id]["layercount"] do
        returntable[i] = table.pack(_G[id]["bias"]["b"..i])
    end
    returntable[#returntable+1] = table.pack(_G[id]["bias"]["ob"])

    return returntable
end

function lnn.debug.returncurrent(id)
    --check for errors.
    lnn.asserttype(id,"id","string")

    if _G[id] == nil then
        error("id ("..id..") doesn't exist.")
    end

    --declare the variables
    local returntable = {}

    --do the stuff
    for i = 1,_G[id]["layercount"] do
        returntable[i] = table.pack(_G[id]["current"]["c"..i])
    end
    returntable[#returntable+1] = table.pack(_G[id]["current"]["o"])

    return returntable
end

function lnn.debug.returngradient(id)
    --check for errors.
    lnn.asserttype(id,"id","string")

    if _G[id] == nil then
        error("id ("..id..") doesn't exist.")
    end

    --do the stuff
    return _G[id]["gradient"]
end

function lnn.debug.returndata(id)
    --check for errors.
    lnn.asserttype(id,"id","string")

    if _G[id] == nil then
        error("id ("..id..") doesn't exist.")
    end

    --do the stuff
    return _G[id]
end

function lnn.debug.clearid(id)
    --check for errors.
    lnn.asserttype(id,"id","string")

    if _G[id] == nil then
        error("id ("..id..") doesn't exist.")
    end

    --do the stuff
    _G[id] = nil
end
