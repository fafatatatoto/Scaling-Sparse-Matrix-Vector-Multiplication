# 這個 project 在做什麼

把 lecture 12 的 sparse matrix-vector multiplication 自己做出來，並且用自己機器上的數字回答投影片上的問題。

一次 SpMV 是 `y = Ax`。稀疏矩陣大部分是 0。每個非零元只做一次乘加（2 個 flop），但要從記憶體讀一個值和一個欄號。投影片的結論是：瓶頸是把矩陣讀進來，算完之後應該被記憶體頻寬卡住，而不是被 CPU 的算力卡住。

這門課後面要你會的三件事，這個 project 都只拿 SpMV 來練：

- 單機上的 locality 和 roofline（lecture 4、5、12）
- 共享記憶體的 OpenMP（lecture 6、12）
- 分散式的 MPI，以及通信要花多少時間（lecture 9、10、11、12）

## 已經做完

對應 lecture 12 的 serial CSR 和 roofline。

在 i5-13420H 上，單執行緒峰值大約 39 GFLOP/s，STREAM 大約 10 GB/s。八個 SuiteSparse 矩陣和一個 dense matvec 的 arithmetic intensity 都在 0.08 到 0.25 FLOP/byte，實際大約 0.9 到 2.1 GFLOP/s。點落在 roofline 的記憶體斜邊上。`scircuit` 比這條線更低。

圖和表：`plots/roofline.png`、`plots/gflops.png`、`results/table1.csv`。

## 接下來的目標

每一項都是投影片上的一個問題。做完要能用一張圖回答。

### 1. OpenMP：列並行為什麼不會一直變快

Lecture 12 用 OpenMP 把每一列分給不同 thread，而且列長不一樣時用 dynamic schedule。投影片問：這樣切為什麼 scale 不好，能不能更好。

要交的東西：同一批矩陣，thread 從 1 加到這台機器的核心數。畫出 speedup。標出停下來的地方是頻寬已經滿了，還是有些列特別長、工作分不均。

### 2. 頻寬滿了以後，少讀一點資料

Lecture 12 說，頻寬吃滿之後，剩下的辦法是減少記憶體流量。課上講的做法是 register blocking：把相鄰非零元收成小的稠密區塊，一個區塊只存一個欄號，區塊裡面不用每個元素都存 index。多出來的 explicit zero 會增加儲存，但迴圈可以展開。

要交的東西：CSR 和 blocked CSR 比一次。看哪些矩陣變快、哪些因為補零變慢。

### 3. MPI：矩陣怎麼切，通信差多少

Lecture 12 先用 dense matvec 暖身，再把同樣三種切法用到 SpMV。Lecture 10 的算法是一次通信要 `τ + nμ`，一個長訊息比很多短訊息便宜。

三種切法：

- 按列切（1D row）：每個 process 有一些列，要用到的 `x` 得跟別人要，或先把 `x` 廣播出去。
- 按欄切（1D column）：每個 process 算出一部分 `y`，最後做一次加總（reduce）。
- 切成二維區塊：廣播和加總只發生在大約 `sqrt(p)` 個 process 上，不是全部。

要交的東西：同一批矩陣、同一組 process 數，三條時間線，每條拆成計算時間和通信時間。

### 4. 把非零元趕進對角區塊

Lecture 12 的理想情況是：reorder 之後，非零元幾乎都在對角區塊裡，區塊外面很少。那樣每個 process 算自己的時候幾乎不用跟別人要 `x`。投影片把這個做法叫 graph partitioning。

要交的東西：reorder 前後，區塊外的非零元佔多少，以及目標 3 的通信時間變了多少。

## 做到哪裡可以放履歷

目標 1 和目標 2 做完，就可以寫共享記憶體上的 SpMV：為什麼慢、加核心為什麼會停、什麼時候少讀資料真的有用。

目標 3 做完，這條就蓋到這門課的 MPI 部分。目標 4 是在目標 3 上面多一個「reorder 讓通信變少」的結果。GPU 是課上的 project 3，用同一批矩陣問一次即可，不必先做。
