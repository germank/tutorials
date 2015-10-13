----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   local avg_loss = 0
   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      avg_loss = avg_loss + criterion:forward(pred, target)
      if confusion then
          confusion:add(pred[{1}] > 0.5 and 2 or 1, target[{1}]+1)
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   print('Average loss: ' .. avg_loss / (testData:size()))
   -- print confusion matrix
   if confusion then
       print(confusion)
       --local p = confusion.mat[2][2] / (confusion.mat[2][1] + confusion.mat[2][2])
       --local r = confusion.mat[2][2] / (confusion.mat[1][2] + confusion.mat[1][2])

       --print ('F1 score: ',2 * p * r / (p+r))

       -- update log/plot
       testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   end
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   if confusion then
      confusion:zero()
   end
end
