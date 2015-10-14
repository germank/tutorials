----------------------------------------------------------------------
-- This script is based on that of Clement Farabet to illustrate
-- how to load data to be used in a supervised model.
--
-- In this case, we are loading the compatibility dataset from 
-- https://github.com/germank/compatibility-naacl2015
--
-- German Kruszewski
----------------------------------------------------------------------

require 'torch'   -- torch

----------------------------------------------------------------------
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Compatibility Data Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-output_type', 'ratings', 'type of data to model: ratings | incompatibility | compatibility')
   cmd:text()
   opt = cmd:parse(arg or {})
end

space_dir = 'data/space/'
-- File containing the word vectors
space_file = 'EN-wform.w.5.cbow.neg10.400.subsmpl_small.th7'
-- Words represented by each of the rows in the space file
space_vocab_file = 'vocab.txt'


-- Datasets to model
if opt.output_type == 'ratings' then
    dataset_dir = 'data/dataset/compatibility-disjoint/'
elseif opt.output_type == 'compatibility' then
    dataset_dir = 'data/dataset/compatibility-disjoint/compatibles/'
elseif opt.output_type == 'incompatibility' then
    dataset_dir = 'data/dataset/compatibility-disjoint/incompatibles/'
else
    error ('Unkown -output_type:  '.. opt.output_type)
end

train_file = 'train_dup.txt'
test_file = 'test_dup.txt'
dev_file = 'dev_dup.txt'

-- Loads the dataset into memory
-- It's stored as a table of tables, such as:
-- {{'baby', 'zebra', 4.1}, {'freezer', 'iguana', 1.0}, ...}
function load_dataset(filename, maxN)
    local ds = {}
    
    local file = io.open(filename)
    for line in file:lines() do
        local get_field = line:gmatch("%S+")
        entry = {}
        table.insert(entry, get_field())
        table.insert(entry, get_field())
        table.insert(entry, tonumber(get_field()))
        
        table.insert(ds, entry)
    end
    return ds
end

trainds = load_dataset(dataset_dir..train_file)
testds = load_dataset(dataset_dir..test_file)
devds = load_dataset(dataset_dir..dev_file)


----------------------------------------------------------------------
-- training/test size

trsize = #trainds
tesize = #testds

----------------------------------------------------------------------
print '==> loading vectors'

-- We load the word vector space from the disk
space = torch.load(space_dir .. space_file)
vector_size = space:size()[2]
word_to_row = {}
local i = 1
for l in io.open(space_dir .. space_vocab_file):lines() do
    word_to_row[l] = i
    i = i + 1
end

print (word_to_row)
-- Takes a loaded dataset and transforms it into an X input matrix
-- by concatenating the words' vectors and copying the dataset ratings
-- into the y output vector
function format_data(ds, space, word_to_row)
    local vector_size = space:size()[2]
    local X = torch.zeros(#ds, vector_size*2)
    local y = torch.zeros(#ds,1)
    for i,entry in pairs(ds) do
        local w1, w2 = entry[1], entry[2]
        -- get the word vectors for the words in the entry
        local w1v, w2v = space[word_to_row[w1]], space[word_to_row[w2]]
        -- copy them concatenated to the X matrix.
        -- (Here we are using slicing. Clement Farbet
        -- wrote a tutorial on slicing available in this same
        -- directory: A_slicing.lua)
        X[{i,{1,vector_size}}]:copy(w1v)
        X[{i,{vector_size+1,2*vector_size}}]:copy(w2v)
        -- copy the ratings to the output vector
        y[i] = entry[3]
    end

    return X, y
end
local trainX, trainy = format_data(trainds, space, word_to_row)

trainData = {
   data = trainX, 
   labels = trainy,
   size = function() return trsize end
}

-- Finally we load the test data.

local testX, testy = format_data(testds, space, word_to_row)
testData = {
   data = testX,
   labels = testy,
   size = function() return tesize end
}

