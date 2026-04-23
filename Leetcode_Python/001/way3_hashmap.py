class Solution(object):
    def twoSum(self,nums,target):

        hashmap = {}

        for i,num in enumerate(nums):
            hashmap[num] = i

        for j in range(len(nums)):
            rest = target - nums[j]
            if (rest in hashmap) and (hashmap[rest] != rest):
                return [j,hashmap[rest]]
        return []
    
# nums = [1,2,3,4,5,6,7,8,9,10]
# target = 19
# test = Solution()
# print(test.twoSum(nums,target))
