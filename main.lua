--READ THE README.md FILE FOR DOCUMENTATION AND OTHER STUFF!!!!
--NOTE: CODE WILL BE IMPROVED LATER AND BUGS WILL BE FIXED

_G["lnn"] = {
	["activation"] = {},
	["loss"] = {},
	["debug"] = {},
	["data"] = {}
}

function lnn.asserttype(variable,variablename,expectedtype)
	--check for errors in the function that checks for errors.
	if type(expectedtype)  ~= "string" or type(variablename) ~= "string" then
		error("variablename and thetype must be a string!")
	end

	--give an error if false or nil
	if type(variable) ~= expectedtype then
		error(string.format("%s (%s) is not a %s or is nil. Type: %s", variablename, tostring(variable), expectedtype, type(variable)))
	end
end

function lnn.assertsize(a,b,aname,bname,zerocheck)
	--check for errors in the function that checks for errors but different.
	if type(a) ~= "table" or type(b) ~= "table" then
		error("a and b must be a table!")
	end
	if type(aname) ~= "string" or type(bname) ~= "string" then
		error("aname and bname must be a string!")
	end

	--give an error they're not the same size or 0.
	if #a ~= #b then
		error(string.format("%s (%s) is not the same size as %s (%s).",aname,#a,bname,#b))
	end

	if zerocheck then
		if #a == 0 or #b == 0 then
			error(string.format("%s (%s) or %s (%s) is equal to zero.",aname,#a,bname,#b))
		end
	end
end

function lnn.findintable(item,table,recursive)
	--check for errors
	lnn.asserttype(table,"table","table")
	lnn.asserttype(recursive,"recursive","boolean")

	--do the stuff
	for i = 1,#table do
		if table[i] == item then
			return i
		end
		if recursive and type(table[i]) == "table" then
			lnn.findintable(item,table[i],true)
		end
	end
	return false
end

function lnn.sumtable(table)
	--check for errors
	lnn.asserttype(table,"table","table")

	--declare the variables
	local sum = 0

	--do the stuff
	for i = 1,#table do
		sum = sum + table[i]
	end

	return sum
end

function lnn.activation.sigmoid(x,derivative)
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

function lnn.activation.tanh(x,derivative)
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

function lnn.activation.relu(x,derivative)
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

function lnn.activation.leakyrelu(x,derivative,alpha)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		if x > 0 then
			return 1
		else
			return alpha
		end
	else
		if x > 0 then
			return x
		else
			return x*alpha
		end
	end
end

function lnn.activation.elu(x,derivative,alpha)
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
			return alpha*(math.exp(x)-1)
		else
			return x
		end
	end
end

function lnn.activation.swish(x,derivative,alpha)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")
	lnn.asserttype(alpha,"alpha","number")

	--do the stuff
	if derivative then
		return x/(1+math.exp(-alpha*x))+lnn.sigmoid(x,false)*(1-x/(1+math.exp(-alpha*x)))
	else
		return x/(1+math.exp(-alpha*x))
	end
end

function lnn.activation.binarystep(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		return 0
	else
		if x > 0 then
			return 1
		else
			return 0
		end
	end
end

function lnn.activation.softmax(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","table")
	lnn.asserttype(derivative,"derivative","boolean")

	--declare the variables
	local expsum = 0
	local returntable = {}

	--do the stuff
	for i = 1,#x do
		expsum = expsum + math.exp(x[i])
	end

	if derivative then
		for i = 1,#x do
			returntable[i] = (math.exp(x[i])/expsum)*(1-(math.exp(x[i])/expsum))
		end
	else
		for i = 1,#x do
			returntable[i] = math.exp(x[i])/expsum
		end
	end
	return returntable
end

function lnn.initialize(id,activation,layersizes)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(activation,"activation","string")
	lnn.asserttype(layersizes,"layersizes","table")

	--check if the id already exists
	if _G[id] ~= nil then
		error(string.format("id (%s) already exists, use the 'lnn.debug.clearid()' to clear the id.",id))
	end

	--check if layersizes has less than 2 items
	if #layersizes < 2 then
		error("layersizes must have at least 2 items for input size and output size. size of layersizes: "..#layersizes)
	end

	--check if the activation is a valid activation function
	if type(lnn.activation[activation]) ~= "function" and activation ~= "linear" then
		error(string.format("%s was not found in lnn.activation, available default activation functions are: 'sigmoid', 'tanh', 'relu', 'leakyrelu', 'elu', 'swish', 'binarystep' and 'linear'.",activation))
	end

	if activation == "softmax" then
		for i = 1,#layersizes-1 do
			if layersizes[i] ~= layersizes[i+1] then
				error("all layers must be an equal size to use the softmax activation function!")
			end
		end
	end

	--initialize the neural network

	--initialize the neural network data
	_G[id] = {
		["activation"] = activation,
		["layercount"] = #layersizes-2,
		["outcount"] = layersizes[#layersizes],
		["insize"] = layersizes[1],
		["alpha"] = 1,
		["gradient"] = {
			["gradw"] = {},
			["gradb"] = {},
			["gwsum"] = 0,
			["gbsum"] = 0,
			["dwinsum"] = 0,
			["learningrate"] = 0
		},
		["id"] = id,
		["weight"] = {},
		["bias"] = {},
		["current"] = {},
		["layersizes"] = layersizes
	}

	if activation == "leakyrelu" then
		_G[id]["alpha"] = 0.01
	end

	--initialize the neural network layers

	local amounttofill = 0

	--create the tables
	for a = 1,#layersizes-1 do
		_G[id]["current"][a] = {}
		_G[id]["bias"][a] = {}
		_G[id]["weight"][a] = {}

		--create the tables for the node values (bias and current)
		for i = 1,layersizes[a+1] do
			_G[id]["current"][a][i] = 0.0
			_G[id]["bias"][a][i] = math.random(-100,100)/100
		end

		--calculate amounttofill
		amounttofill = layersizes[a+1]*layersizes[a]

	    --create the tables for the connection values (weight)
		for i = 1,amounttofill do
			_G[id]["weight"][a][i] = math.random(-100,100)/100
		end
	end
end

function lnn.forwardpass(id,intable)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")

	--check if the id doesn't exist
	if _G[id] == nil then
		error(string.format("id (%s) doesnt exist.",id))
	end

	if #intable ~= _G[id]["insize"] then
		error(string.format("intable (%s) is not the same size as the intable when id (%s) was initialized (%s).",#intable,id,_G[id]["insize"]))
	end

	--declare the functions
	local function getlayer(lastlayer,nextlayer,weights,biases) --i am super proud of this function :D
		--declare the variables
		local sum = 0

		--get the sum of the connected weights to the current node we are on and replace the nextlayer
		if type(lnn.activation[_G[id]["activation"]]) == "function" then --is this inefficient? yes.
			for a = 1,#nextlayer do
				for i = 1,#lastlayer do
					sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
				end
				nextlayer[a] = lnn.activation[_G[id]["activation"]](sum+biases[a],false,_G[id]["alpha"]) --even if it only has 2 parameters this still works but it pains me to do that
				sum = 0
			end
		elseif _G[id]["activation"] == "linear" then
			for a = 1,#nextlayer do
				for i = 1,#lastlayer do
					sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
				end
				nextlayer[a] = (sum+biases[a])*_G[id]["alpha"]
				sum = 0
			end
		else
			error(string.format("id %s has an invalid activation function! (%s)",id,_G[id]["activation"]))
		end
	end

	--do the stuff
	getlayer(intable,_G[id]["current"][1],_G[id]["weight"][1],_G[id]["bias"][1]) --input layer to first hidden
	for i = 2,_G[id]["layercount"]+1 do --rest of the hidden layers
		getlayer(_G[id]["current"][i-1],_G[id]["current"][i],_G[id]["weight"][i],_G[id]["bias"][i])
	end

	return _G[id]["current"][#_G[id]["current"]] --last table in current is output layer
end

function lnn.adjust(id,intable,output,expectedoutput,learningrate)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")
	lnn.asserttype(learningrate,"learningrate","number")

	lnn.assertsize(output,expectedoutput,"out","expectedout",true)
	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end
	if _G[id]["insize"] ~= #intable then
		error(string.format("insize (%s) is not the same size as intable (%s).",_G[id]["insize"],#intable))
	end

	--declare the variables
	local grad = {
		["gradw"] = {},
		["gradb"] = {},
		["gwsum"] = 0, --weight gradient sum (for hidden layers)
		["gbsum"] = 0, --bias gradient sum (for hidden layers aswell)
		["dwinsum"] = 0, --derivative of the weighted sum of the input
		["learningrate"] = learningrate
	}

	--get the sum of the weighted inputs
	for a = 1,#_G[id]["current"][1] do
		for i = 1,#intable do
			grad["dwinsum"] = grad["dwinsum"] + _G[id]["weight"][1][i+((a-1)*#intable)]*intable[i]
		end
	end

	if type(lnn.activation[_G[id]["activation"]]) == "function" then
		grad["dwinsum"] = lnn.activation[_G[id]["activation"]](grad["dwinsum"],true,_G[id]["alpha"])
	elseif _G[id]["activation"] == "linear" then
		grad["dwinsum"] = 1
	else
		error(string.format("id %s has an invalid activation function, activation: %s",id,_G[id]["activation"]))
	end

	--get gradw
	for i = 1,#output do
		grad["gradw"][i] = ((output[i]-expectedoutput[i])^2*grad["dwinsum"])*learningrate+((output[i]-expectedoutput[i])*learningrate)
	end

	--get gradb
	for i = 1,#output do
		grad["gradb"][i] = (output[i]-expectedoutput[i])*learningrate+((output[i]-expectedoutput[i])*learningrate)
	end

	--get gradwsum
	for i = 1,#output do
		grad["gwsum"] = grad["gwsum"] + grad["gradw"][i]
	end

	--get gradbsum
	for i = 1,#output do
		grad["gbsum"] = grad["gbsum"] + grad["gradb"][i]
	end

	--update the data on _G[id]
	_G[id]["gradient"] = grad

	--adjust the output layer weights and biases
	for a = 1,#output do
		--adjust the output layer weights
		for i = 1,#_G[id]["current"][_G[id]["layercount"]] do
			_G[id]["weight"][#_G[id]["weight"]][i+((a-1)*#_G[id]["current"][_G[id]["layercount"]])] = _G[id]["weight"][#_G[id]["weight"]][i+((a-1)*#_G[id]["current"][_G[id]["layercount"]])] - grad["gradw"][i]
		end

		--adjust the output layer biases
		_G[id]["bias"][#_G[id]["bias"]][a] = _G[id]["bias"][#_G[id]["bias"]][a] - grad["gradb"][a]
	end

	--adjust the rest of the weights and biases
	for b = _G[id]["layercount"],1,-1 do
		--adjust the hidden layer biases
		for i = 1,#_G[id]["bias"][b] do
			_G[id]["bias"][b][i] = _G[id]["bias"][b][i] - grad["gbsum"]
		end

		--adjust the hidden layer weights
		for i = 1,#_G[id]["weight"][b] do
			_G[id]["weight"][b][i] = _G[id]["weight"][b][i] - grad["gwsum"]
		end
	end
end

function lnn.returngradient(id,intable,output,expectedoutput,learningrate)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")
	lnn.asserttype(learningrate,"learningrate","number")

	lnn.assertsize(output,expectedoutput,"out","expectedout",true)
	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end
	if _G[id]["insize"] ~= #intable then
		error(string.format("insize (%s) is not the same size as intable (%s).",_G[id]["insize"],#intable))
	end

	--declare the variables
	local grad = {
		["gradw"] = {},
		["gradb"] = {},
		["gwsum"] = 0, --weight gradient sum (for hidden layers)
		["gbsum"] = 0, --bias gradient sum (for hidden layers aswell)
		["dwinsum"] = 0, --derivative of the weighted sum of the inputs
		["learningrate"] = learningrate
	}

	--get the sum of the weighted inputs
	for a = 1,#_G[id]["current"][1] do
		for i = 1,#intable do
			grad["dwinsum"] = grad["dwinsum"] + _G[id]["weight"][1][i+((a-1)*#intable)]*intable[i]
		end
	end

	if type(lnn.activation[_G[id]["activation"]]) == "function" then
		grad["dwinsum"] = lnn.activation[_G[id]["activation"]](grad["dwinsum"],true,_G[id]["alpha"])
	elseif _G[id]["activation"] == "linear" then
		grad["dwinsum"] = 1
	else
		error(string.format("id %s has an invalid activation function, activation: %s",id,_G[id]["activation"]))
	end

	--get gradw
	for i = 1,#output do
		grad["gradw"][i] = ((output[i]-expectedoutput[i])^2*grad["dwinsum"])*learningrate+((output[i]-expectedoutput[i])*learningrate)
	end

	--get gradb
	for i = 1,#output do
		grad["gradb"][i] = (output[i]-expectedoutput[i])*learningrate+((output[i]-expectedoutput[i])*learningrate)
	end

	--get gradwsum
	for i = 1,#output do
		grad["gwsum"] = grad["gwsum"] - grad["gradw"][i]
	end

	--get gradbsum
	for i = 1,#output do
		grad["gbsum"] = grad["gbsum"] - grad["gradb"][i]
	end

	--update the data on _G[id]
	_G[id]["gradient"] = grad

	return grad
end

function lnn.adjustfromgradient(id,gradient)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(gradient,"gradient","table")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	--update the data on _G[id]
	_G[id]["gradient"] = gradient

	--adjust the output layer weights and biases
	for a = 1,#_G[id]["outcount"] do
		--adjust the output layer weights
		for i = 1,#_G[id]["current"][_G[id]["layercount"]] do
			_G[id]["weight"][#_G[id]["weight"]][i+((a-1)*#_G[id]["current"][_G[id]["layercount"]])] = _G[id]["weight"][#_G[id]["weight"]][i+((a-1)*#_G[id]["current"][_G[id]["layercount"]])] - gradient["gradw"][i]
		end

		--adjust the output layer biases
		_G[id]["bias"][#_G[id]["bias"]][a] = _G[id]["bias"][#_G[id]["bias"]][a] - gradient["gradb"][a]
	end

	--adjust the rest of the weights and biases
	for b = _G[id]["layercount"],1,-1 do
		--adjust the hidden layer biases
		for i = 1,#_G[id]["bias"][b] do
			_G[id]["bias"][b][i] = _G[id]["bias"][b][i] - gradient["gbsum"]+((_G[id]["bias"][b][i]*-gradient["gbsum"])*gradient["learningrate"])
		end

		--adjust the hidden layer weights
		for i = 1,#_G[id]["weight"][b] do
			_G[id]["weight"][b][i] = _G[id]["weight"][b][i] - gradient["gwsum"]+((_G[id]["weight"][b][i]*-gradient["gwsum"])*gradient["learningrate"])
		end
	end
end

function lnn.loss.mse(output,expectedoutput)
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

function lnn.loss.mae(output,expectedoutput)
	--check for errors
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedutput","table")

	lnn.assertsize(output,expectedoutput,"output","expectedoutput")

	--declare the variables
	local mae = 0

	--do the stuff
	for i = 1,#output do
		mae = mae + math.abs(output[i] - expectedoutput[i])
	end
	return mae/#output
end

function lnn.loss.sse(output,expectedoutput)
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

function lnn.loss.rmse(output,expectedoutput)
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

function lnn.loss.crossentropy(output,expectedoutput)
	--check for errors.
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")

	lnn.assertsize(output,expectedoutput,"output","expectedoutput",true)

	--declare the variables
	local sum = 0

	--do the stuff
	for i = 1,#output do
		sum = sum + (expectedoutput[i]*math.log(output[i]+0.01)) + (1-expectedoutput[i]) * math.log(1-output[i])
	end
	return -sum
end

--either debugging or visualizing, could be used for both.

function lnn.debug.returnweights(id)
	--check for errors.
	lnn.asserttype(id,"id","string")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	return _G[id]["weight"]
end

function lnn.debug.returnbiases(id)
	--check for errors.
	lnn.asserttype(id,"id","string")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	return _G[id]["bias"]
end

function lnn.debug.returncurrent(id)
	--check for errors.
	lnn.asserttype(id,"id","string")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	return _G[id]["current"]
end

function lnn.debug.clearid(id)
	--check for errors.
	lnn.asserttype(id,"id","string")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	--do the stuff
	_G[id] = nil
end

function lnn.debug.randomize(id)
	--check for errors
	lnn.asserttype(id,"id","string")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	--do the stuff

	for a = 1,_G[id]["layercount"]+1 do
		--randomize biases
		for i = 1,#_G[id]["bias"][a] do
			_G[id]["bias"][a][i] = math.random(-100,100)/100
		end

		--randomize weights
		for i = 1,#_G[id]["weight"][a] do
			_G[id]["weight"][a][i] = math.random(-100,100)/100
		end
	end
end

function lnn.debug.addrandom(id,lowerlimit,upperlimit)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(lowerlimit,"lowerlimit","number")
	lnn.asserttype(upperlimit,"upperlimit","number")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	--do the stuff
	for a = 1,_G[id]["layercount"]+1 do
		--randomize biases
		for i = 1,#_G[id]["bias"][a] do
			_G[id]["bias"][a][i] = _G[id]["bias"][a][i] + math.random(lowerlimit,upperlimit)
		end

		--randomize weights
		for i = 1,#_G[id]["weight"][a] do
			_G[id]["weight"][a][i] = _G[id]["weight"][a][i] + math.random(lowerlimit,upperlimit)
		end
	end
end

--data, used for importing and exporting commonly used data. is this possibly insecure? yes !

function lnn.data.exportdata(id,filename)
	--check for errors.
	lnn.asserttype(id,"id","string")
	lnn.asserttype(filename,"filename","string")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	--clear the file
	io.open(filename,"w"):close()

	--do the stuff
	local f = io.open(filename,"a+")

	if f ~= nil then
		--write the basic data
		f:write(string.format("%s = {\n['activation'] = '%s',\n['layercount'] = %s,\n['outcount'] = %s,\n['insize'] = %s,\n['alpha'] = %s,\n['id'] = '%s',\n['weight'] = {\n",id,_G[id]["activation"],_G[id]["layercount"],_G[id]["outcount"],_G[id]["insize"],_G[id]["alpha"],id))

		--write the weight data
		for a = 1,_G[id]["layercount"]+1 do
			f:write("{")
			for i = 1,#_G[id]["weight"][a] do
				f:write(_G[id]["weight"][a][i]..",")
			end
			f:write("},\n")
		end

		f:write("},\n['bias'] = {\n")

		--write the bias data
		for a = 1,_G[id]["layercount"]+1 do
			f:write("{")
			for i = 1,#_G[id]["bias"][a] do
				f:write(_G[id]["bias"][a][i]..",")
			end
			f:write("},\n")
		end

		f:write("},\n['current'] = {\n")

		--write the current data
		for a = 1,_G[id]["layercount"]+1 do
			f:write("{")
			for i = 1,#_G[id]["current"][a] do
				f:write(_G[id]["current"][a][i]..",")
			end
			f:write("},\n")
		end

		f:write(string.format("}\n}\nreturn %s",id))
	end
end

function lnn.data.importdata(id,filename)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(filename,"filename","string")

	if _G[id] ~= nil then
		error(string.format("id %s already exists, use lnn.debug.clearid(%s) to clear the id.",id,id))
	end
	--do the stuff
	local f = io.open(filename,"r")
	if f ~= nil then
		_G[id] = load(f:read(f:seek("end",0)))()
	else
		error("filename is nil, maybe it doesn't exist?")
	end
end
