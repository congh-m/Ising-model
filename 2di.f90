 program main1
     parameter(n=91)				!!!!!!!!!!!!!!!!!!1
     integer i,j,tmax,ti,nn,bin,bins
     common a(n,n)
     real m,z,t,zi
     real(KIND=8) :: ran
     parameter(tmax=40)

	bins=100		!设置统计平均bin的次数
 	nn=int(n*n*1.3)		!定义适宜的预热次数


     ! 设置输出文件
     open(1,file='out-n91.dat')		!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
     open(2,file='n91.dat')

     ! 初始化Ising格点，均赋值为1 
     do 10 i=1,n
     do 20 j=1,n
         a(i,j)=1
     20  continue
     10  continue
   

     ! 体系按温度改变(0.1~4.0)进行Monte Carlo模拟
     do 30 ti=10,tmax,1
	t=ti*0.1

     ! 体系的预热阶段，在此期间不采样
      do 40,i=1,nn	
        call mcstep(t) 		! 调用metropolis算法子例程
      40  continue

     ! 预热结束，可以对体系进行采样了

      m = 0.0			! 为每个温度点的磁化强度初始化
      do 50 bin=1,bins
      call mcstep(t)
      zi=0.0			! 为每个bin统计磁矩初始化

      do 60,k=1,nn		! 对体系进行采样
        call mcstep(t)		
        z=0.0			! 为每次抽样的磁化强度计算做初始化

        do 70,i=1,n
        do 80,j=1,n
        z=z+a(i,j)		! 统计一次采样的总磁矩朝向
	80     continue
	70     continue

        z=abs(z)/(n*n)		! 计算一次抽样的磁化强度
        zi=z/nn+zi		! 将所有抽样的磁化强度做统计平均，即一个bin的磁化强度

      60   continue
      
      m=zi/bins+m		! 将所有bin的磁化强度做统计平均
      50   continue

      write(1,200)t,m 		! 记录M随着T的变化曲线，向文件输出
      write(*,200)t,m 		! 向屏幕输出
      write(2,300)m		! 单独记录M

     30   continue
   
     200  format(2x,f6.4,8x,f10.6)	! 定义输出数据“200”的格式
     300  format(2x,f10.6)	! 定义输出数据“300”的格式

       close(1)			!关闭代码为1的文件夹
       close(2)			!关闭代码为2的文件夹
      end			!主程序结束


 
     ! 实现子程序Metropolis算法 
      subroutine mcstep(t)
      parameter(n=91)				!!!!!!!!!!!!!!!!!!!!!!!!!
      common a(n,n)
      do 10,i=1,n
      do 20,j=1,n
 
     ! x1（上）, x2（下）, x3（左）, x4（右） 是当前格点(i,j)的四个近邻
       x1=a(i-1,j)
       x2=a(i+1,j)
       x3=a(i,j-1)
       x4=a(i,j+1)
       if(i.eq.1) x1=a(n,j)		! 定义边界条件，第一行和最后一行连起来
       if(i.eq.n) x2=a(1,j)		! 最后一行连到第一行
       if(j.eq.1) x3=a(i,n)		! 第一列和最后一列连起来
       if(j.eq.n) x4=a(i,1)		! 最后一列连到第一列
 
     ! h表示体系尝试翻转前后的Hamiltonian(能量)变化
       h=a(i,j)*(x1+x2+x3+x4)
       w=exp(-2*h/t)
       b=rand()

     ! 按Metropolis规则，判断是否接受当前格点的尝试翻转
       if(w.gt.b) a(i,j)=-a(i,j)
       if(w.le.b) a(i,j)=a(i,j)
       
      20   continue
      10   continue
        end				!子程序Metropolis结束
  
     ! 产生0~1均匀分布的随机数
      function rand()
      real rand
      integer seed,c1,c2,c3
      parameter(c1=29,c2=217,c3=1024)
      data seed/1/
      seed=mod(seed*c1+c2,c3)
      rand=real(seed)/c3		!返回随机数的值
      end				!随机数函数结束


