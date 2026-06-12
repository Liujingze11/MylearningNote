class Solution(object):
    def permute(self, nums):
        result = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            print(f"调用 backtrack，当前 path：{path}")
            if len(path) == len(nums):
                print(f"✅ 找到一个排列：{path}")
                result.append(path[:])  # 注意拷贝
                return
            
            for i in range(len(nums)):
                if used[i]:
                    continue

                # 做选择
                print(f"选择 nums[{i}] = {nums[i]}")
                path.append(nums[i])
                used[i] = True

                # 递归
                backtrack()

                # 撤销选择（回溯）
                print(f"回溯，撤销选择 nums[{i}] = {nums[i]}")
                path.pop()
                used[i] = False

        backtrack()
        return result
sol = Solution()
sol.permute([1, 2, 3])
