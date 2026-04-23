class Solution(object):
    def twoSum(self,nums,target):
        for i in range(lens(nums)):
            for j in range(i+1,lens(nums)):
                if nums[i] + nums[j] == target:
                    return [i,j]