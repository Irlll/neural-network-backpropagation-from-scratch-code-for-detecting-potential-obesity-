# Math Behind Multilayer Perceptron Neural Networks Backpropagation with Manual Code Python and Excel for Detecting Potential Obesity (Step by Step Indonesia)

#### Disusun oleh : Irbah Labibah Nur Saidah

<br>

## Backpropagation Neural Network

Backpropagation termasuk ke dalam neural network dan merupakan metode pelatihan terawasi (supervised learning) menggunakan jaringan multilayer, yang khusus meminimalkan kesalahan output yang dihasilkan oleh jaringan Backpropagation mengevaluasi kontribusi kesalahan dari
setiap neuron setelah satu set data diproses. Tujuan backpropagation adalah untuk memodifikasi bobot dengan melatih jaringan neural sehingga dapat memprediksi output dengan benar.

Perceptron multilayer dapat dilatih menggunakan algoritma backpropagasi. Tujuannya adalah untuk mempelajari bobot untuk semua keterkaitan dalam jaringan berlapis-lapis. Minimum fungsi kesalahan dalam ruang bobot dihitung menggunakan metode penurunan gradien. Bobot resultan yang menawarkan fungsi kesalahan minimum merupakan solusi dari masalah pembelajaran.

Dalam prosesnya, backpropagation berkerja dengan cara melakukan dua tahap perhitungan yaiu perhitungan maju (forward pass) yang akan menghitung nlai kesalahan (error) antara nilai output sistem dengan nilai yang seharusnya dan perhitungan mundur (backward pass) untuk memperbaiki bobot berdasarkan nilai error tersebut.

<br>

## Gradient Descent

Penurunan gradient (gradient descent) adalah algoritma optimasi iteratif untuk menemukan fungsi minimum. Pada backpropagation, penurunan gradient digunakan untuk memperbarui (update) bobot sehingga dapat meminimalkan fungsi kesalahan.

  -----------------------------------------------------------------------
  $$Wxnew = Wx - a*\frac{\partial Error}{\partial Wx}$$
  -----------------------------------------------------------------------

  $$Wxnew = New\ weight$$
  
  $$Wx = Old\ weight$$

  $$a = Learning\ rate$$
  
  -----------------------------------------------------------------------
  

<br>

## Fungsi Aktivasi Sigmoid

Disebut sigmoid karena berbentuk S. Fungsi ini dapat dilihat sebagai menekan nilai argumennya ke dalam rentang {0,1} . Sigmoid mengambil bilangan real apa pun, dan menghasilkan angka antara 0 dan 1.

<br>

<div><img src="https://user-images.githubusercontent.com/107544829/190966980-4c26e9aa-653f-48bd-9d5e-0a26a7a7157e.png" width="300"/></div>

<br>

## Detecting Potential Obesity

Rancangan sistem ini mengimplementasi metode multilayer perceptron
backpropagation, menggunakan gradient descent untuk mengupdate bobot,
dan menggunakan sigmoid untuk fungsi aktivasi.

Rancangan sistem deteksi potensi obesitas ini memiliki 1 output, yaitu 0
jika tidak berpotensi obesitas atau 1 jika berpotensi obesitas.

Rancangan sistem deteksi potensi obesitas ini memiliki 4 input sebagai
berikut :

1. Apakah ada anggota keluarga yang menderita atau menderita kelebihan
    berat badan? (x1)

    a.  Ya (1)

    b.  Tidak (2)

2. Apakah Anda sering makan makanan berkalori tinggi? (x2)

    a.  Ya (1)

    b.  Tidak (2)

3. Berapa banyak makanan utama yang Anda makan setiap hari? (x3)

    a.  Satu atau dua (1)

    b.  Tiga (3)

    c.  Lebih dari tiga (4)

4. Seberapa sering Anda rutin melakukan aktivitas fisik dalam seminggu?
    (x4)

    a.  Saya tidak rutin melakukan aktivitas fisik (1)

    b.  Sekitar satu sampai dua hari (2)

    c.  Sekitar dua sampai empat hari (3)

    d.  Sekitar empat sampai lima hari (4)

<br>

Dataset yang didapat :

<br>

<div><img src="https://user-images.githubusercontent.com/107544829/190967325-713661f8-bf57-4dbd-b571-517ce48a3a5a.png" width="500"/></div>

<br>

Berdasarkan rancangan tersebut, berikut arsitektur jaringan deteksi potensi obesitas dengan 4 input dan 1 output :

<br>

<div><img src="https://user-images.githubusercontent.com/107544829/190967523-9cfafc5a-aa5a-4d2d-abfc-2e77d292241b.png" width="500"/></div>

<br>

Arsitektur ini merupakan arsitektur multilayer perceptron dengan 1 hidden layer.

Dalam prosesnya, backpropagation berkerja dengan cara melakukan 2 tahap perhitungan yaiu perhitungan maju (forward pass) yang akan menghitung nlai kesalahan (error) antara nilai output sistem dengan nilai yang seharusnya dan perhitungan mundur (backward pass) untuk memperbaiki bobot berdasarkan nilai error tersebut.

<br>

## Backpropagation Formula

### Forward Pass

#### `Langkah 1` : Menghitung setiap net pada hidden layer

neth1 = w11 \* x1 + w12 \* x2 + w13 \* x3 + w14 \* x4 + biash1

neth2 = w21 \* x1 + w22 \* x2 + w23 \* x3 + w24 \* x4 + biash2

neth3 = w31 \* x1 + w32 \* x2 + w33 \* x3 + w34 \* x4 + biash3

#### `Langkah 2` : Menghitung setiap output pada hidden layer dengan

menggunakan fungsi aktivasi sigmoid

oh 1 = $\frac{1}{1 + e^{- neth1}}$

oh 2 = $\frac{1}{1 + e^{- neth2}}$

oh 3 = $\frac{1}{1 + e^{- neth3}}$

#### `Langkah 3` : Menghitung net pada output layer

netout = wh1 \* oh1 + wh2 \* oh2 + wh3 \* oh3 + biasout

#### `Langkah 4` : Menghitung output pada output layer dengan menggunakan

fungsi aktivasi sigmoid

out = $\frac{1}{1 + e^{- netout}}$

#### `Langkah 5` : Menghitung error pada output layer

E = (target - out)<sup>2</sup>

<br>

### Backward Pass

#### `Langkah 6` : Update bobot di output layer

Misalkan update bobot wh1

wh<sup>+</sup> = wh - $\eta$ \* $\frac{\partial E}{\partial wh}$

$\frac{\partial E}{\partial wh1}$ = $\frac{\partial E}{\partial out}$ \*
$\frac{\partial out}{\partial wh1}$

= $\frac{\partial E}{\partial out}$ \*
$\frac{\partial out}{\partial netout}$ \*
$\frac{\partial netout}{\partial wh1}$

- E = (target - out)<sup>2</sup>

  $\frac{\partial E}{\partial out}$ = 2 \* (target - out)<sup>2-1</sup> \* -1

  = -2 \* (target - out)

- out = $\frac{1}{1 + e^{- netout}}$

    $\frac{\partial out}{\partial netout}$ = out \* (1 - out)

- netout = wh1 \* oh1 + wh2 \* oh2 + wh3 \* oh3 + biasout

    $\frac{\partial netout}{\partial wh1}$ = oh1

$\frac{\partial E}{\partial wh1}$ =
$\frac{\partial E}{\partial out}$ \*
$\frac{\partial out}{\partial netout}$ \*
$\frac{\partial netout}{\partial wh1}$

= (-2 \* (target - out)) \* (out \* (1 - out)) \* (oh1)

wh1<sup>+</sup> = wh1 - $\eta$ \* $\frac{\partial E}{\partial wh1}$

#### Langkah 7 : Update bias di output layer

biasout<sup>+</sup> = biasout - $\eta$ \*
$\frac{\partial E}{\partial biasout}$

$\frac{\partial E}{\partial biasout}$ =
$\frac{\partial E}{\partial out}$ \*
$\frac{\partial out}{\partial netout}$ \*
$\frac{\partial netout}{\partial biasout}$

- netout = wh1 \* oh1 + wh2 \* oh2 + wh3 \* oh3 + biasout

    $\frac{\partial netout}{\partial biasout}$ = 0 + 0 + 0 + 1

    = 1

$\frac{\partial E}{\partial biasout}$ =
$\frac{\partial E}{\partial out}$ \*
$\frac{\partial out}{\partial netout}$ \*
$\frac{\partial netout}{\partial biasout}$

= (-2 \* (target - out)) \* (out \* (1 - out)) \* (1)

biasout<sup>+</sup> = biasout - $\eta$ \*
$\frac{\partial E}{\partial biasout}$

#### `Langkah 8` : Update bobot di hidden layer

Misalkan update bobot w11

w<sup>+</sup> = w - $\eta$ \* $\frac{\partial E}{\partial w}$

$\frac{\partial E}{\partial w11}$ =
$\frac{\partial E}{\partial oh1}$ \*
$\frac{\partial oh1}{\partial neth1}$ \*
$\frac{\partial neth1}{\partial w11}$

= $\frac{\partial E}{\partial netout}$ \*
$\frac{\partial netout}{\partial oh1}$ \*
$\frac{\partial oh1}{\partial neth1}$ \*
$\frac{\partial neth1}{\partial w11}$

- $\frac{\partial E}{\partial netout}$ =
    $\frac{\partial E}{\partial out}$ \*
    $\frac{\partial out}{\partial netout}$ = (-2 \* (target - out)) \*
    (out \* (1 - out))

- netout = oh1 \* wh1\* + oh2 \* wh2 + oh3 \* wh3 + biasout

    $\frac{\partial netout}{\partial oh1}$ = wh1

- oh 1 = $\frac{1}{1 + e^{- neth1}}$

    $\frac{\partial oh1}{\partial neth1}$ = oh1 \* (1 - oh1)

- neth1 = w11 \* x1 + w12 \* x2 + w13 \* x3 + w14 \* x4 + biash1

    $\frac{\partial neth1}{\partial w11}$ = x1

$\frac{\partial E}{\partial w11}$ = ((-2 \* (target - out)) \* (out
    \* (1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x1)

w11<sup>+</sup> = w11 - $\eta$ \* $\frac{\partial E}{\partial w11}$

#### `Langkah 9` : Update bias di hidden layer

Misalkan update bobot pada biash1

biash<sup>+</sup> = biash - $\eta$ \* $\frac{\partial E}{\partial biash}$

$\frac{\partial E}{\partial biash1}$ =
$\frac{\partial E}{\partial oh1}$ \*
$\frac{\partial oh1}{\partial neth1}$ \*
$\frac{\partial neth1}{\partial biash1}$

= $\frac{\partial E}{\partial netout}$ \*
$\frac{\partial netout}{\partial oh1}$ \*
$\frac{\partial oh1}{\partial neth1}$ \*
$\frac{\partial neth1}{\partial biash1}$

- neth1 = w11 \* x1 + w12 \* x2 + w13 \* x3 + w14 \* x4 + biash1

    $\frac{\partial neth1}{\partial biash1}$ = 0 + 0 + 0 + 1

    = 1

$\frac{\partial E}{\partial biash1}$ =
$\frac{\partial E}{\partial netout}$ \*
$\frac{\partial netout}{\partial oh1}$ \*
$\frac{\partial oh1}{\partial neth1}$ \*
$\frac{\partial neth1}{\partial biash1}$

= ((-2 \* (target - out)) \* (out \* (1 - out))) \* (wh1) \* (oh1
    \* (1 - oh1)) \* (1)

biash1<sup>+</sup> = biash1 - $\eta$ \* $\frac{\partial E}{\partial biash1}$

<br>

## Calculating Backpropagation (Input 1 - 3)

### Input 1

> x1 = 2; x2 = 2; x3 = 3; x4 = 3; Target = 0

Bobot awal hidden layer :

> w11 = 0.05; w12 = 0.1; w13 = 0.15; w14 = 0.2; biash1 = 0.25

> w21 = 0.3; w22 = 0.35; w23 = 0.4; w24 = 0.45; biash2 = 0.5

> w31 = 0.55; w32 = 0.6; w33 = 0.65 w34 = 0.7 biash3 = 0.75

Bobot awal output layer :

wh1= 0.8; wh2 = 0.85; wh3 = 0.9

#### Forward Pass

`Langkah 1` :

neth1 = w11 \* x1 + w12 \* x2 + w13 \* x3 + w14 \* x4 + biash1

= 0.05 \* 2 + 0.1 \* 2 + 0.15 \* 3 + 0.2 \* 3 + 0.25

= 1.6

neth2 = w21 \* x1 + w22 \* x2 + w23 \* x3 + w24 \* x4 + biash2

= 0.3 \* 2 + 0.35 \* 2 + 0.4 \* 3 + 0.45 \* 3 + 0.5

= 4.35

neth3 = w31 \* x1 + w32 \* x2 + w33 \* x3 + w34 \* x4 + biash3

= 0.55 \* 2 + 0.6 \* 2 + 0.65 \* 3 + 0.7 \* 3 + 0.75

= 7.1

<br>

`Langkah 2` :

oh 1 = $\frac{1}{1 + e^{- neth1}}$

= $\frac{1}{1 + e^{- 1.6}}$

= 0.8320183851

oh 2 = $\frac{1}{1 + e^{- neth2}}$

= $\frac{1}{1 + e^{- 4.35}}$

= 0.9872576505

oh 3 = $\frac{1}{1 + e^{- neth3}}$

= $\frac{1}{1 + e^{- 7.1}}$

= 0.9991755753

<br>

`Langkah 3` :

netout = wh1 \* oh1 + wh2 \* oh2 + wh3 \* oh3 + biasout

= 0.8 \* 0.8320183851 + 0.85 \* 0.9872576505 + 0.9 \* 0.9991755753 +
0.95

= 3.354041729

<br>

`Langkah 4` :

out = $\frac{1}{1 + e^{- netout}}$

= $\frac{1}{1 + e^{- 3.354041729}}$

= 0.9662369384

Langkah 5 :

E = (target - out)<sup>2</sup>

= (0 - 0.9662369384)<sup>2</sup>

= 0.9336138211

#### Backward Pass

`Langkah 6` :

$\frac{\partial E}{\partial wh1}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh1)

= (-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)) \*
(0.8320183851)

= 0.05245320288

wh1<sup>+</sup> = wh1 - $\eta$ \* $\frac{\partial E}{\partial wh1}$

= 0.8 - 0.5 \* 0.05245320288

= 0.7737733985

$\frac{\partial E}{\partial wh2}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh2)

= (-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)) \*
(0.9872576505)

= 0.06224000187

wh2<sup>+</sup> = wh2 - $\eta$ \* $\frac{\partial E}{\partial wh2}$

= 0.85 - 0.5 \* 0.06224000187

= 0.8188799991

$\frac{\partial E}{\partial wh3}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh3)

= (-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)) \*
(0.9991755753)

= 0.06299134744

wh3<sup>+</sup> = wh3 - $\eta$ \* $\frac{\partial E}{\partial wh3}$

= 0.9 - 0.5 \* 0.06299134744

= 0.8685043263

<br>

`Langkah 7` :

$\frac{\partial E}{\partial biasout}$ = (-2 \* (target - out)) \* (out
\* (1 - out)) \* (1)

= (-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)) \*
(1)

= 0.06304332191

biasout<sup>+</sup> = biasout - $\eta$ \* $\frac{\partial E}{\partial biasout}$

= 0.95 - 0.5 \* 0.06304332191

= 0.918478339

<br>

`Langkah 8` :

$\frac{\partial E}{\partial w11}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x1)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
0.8 \* (0.8320183851 \* (1 - 0.8320183851)) \* (2)

= 0.01409787796

w11<sup>+</sup> = w11 - $\eta$ \* $\frac{\partial E}{\partial w11}$

= 0.05 - 0.5 \*0.01409787796

= 0.04295106102

$\frac{\partial E}{\partial w12}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x2)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
0.8 \* (0.8320183851 \* (1 - 0.8320183851)) \* (2)

= 0.01409787796

w12<sup>+</sup> = w12 - $\eta$ \* $\frac{\partial E}{\partial w12}$

= 0.1 - 0.5 \* 0.01409787796

= 0.09295106102

$\frac{\partial E}{\partial w13}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x3)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)))
\* 0.8 \* (0.8320183851 \* (1 - 0.8320183851)) \* (3)

= 0.02114681695

w13<sup>+</sup> = w13 - $\eta$ \* $\frac{\partial E}{\partial w13}$

= 0.15 - 0.5 \* 0.02114681695

= 0.1394265915

$\frac{\partial E}{\partial w14}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x4)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)))
\* 0.8 \* (0.8320183851 \* (1 - 0.8320183851)) \* (3)

= 0.02114681695

w14<sup>+</sup> = w14 - $\eta$ \* $\frac{\partial E}{\partial w14}$

= 0.2 - 0.5 \* 0.02114681695

= 0.1894265915

$\frac{\partial E}{\partial w21}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x1)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
(0.85) \* (0.9872576505 \* (1 - 0.9872576505)) \* (2)

= 0.001348242556

w21<sup>+</sup> = w21 - $\eta$ \* $\frac{\partial E}{\partial 21}$

= 0.3 - 0.5 \* 0.001348242556

= 0.2993258787

$\frac{\partial E}{\partial w22}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x2)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)))
\* (0.85) \* (0.9872576505 \* (1 - 0.9872576505)) \* (2)

= 0.001348242556

w22<sup>+</sup> = w22 - $\eta$ \* $\frac{\partial E}{\partial w22}$

= 0.35 - 0.5 \* 0.001348242556

= 0.3493258787

$\frac{\partial E}{\partial w23}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x3)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)))
\* (0.85) \* (0.9872576505 \* (1 - 0.9872576505)) \* (3)

= 0.002022363834

w23<sup>+</sup> = w23 - $\eta$ \* $\frac{\partial E}{\partial w23}$

= 0.4 - 0.5 \* 0.002022363834

= 0.3989888181

$\frac{\partial E}{\partial w24}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x4)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)))
\* (0.85) \* (0.9872576505 \* (1 - 0.9872576505)) \* (3)

= 0.002022363834

w24<sup>+</sup> = w24 - $\eta$ \* $\frac{\partial E}{\partial w24}$

= 0.45 - 0.5 \* 0.002022363834

= 0.4489888181

$\frac{\partial E}{\partial w31}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x1)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
(0.9) \* (0.9991755753 \* (1 - 0.9991755753)) \* (2)

= 0.00009347692088

w31<sup>+</sup> = w31 - $\eta$ \* $\frac{\partial E}{\partial 31}$

= 0.55 - 0.5 \* 0.00009347692088

= 0.5499532615

$\frac{\partial E}{\partial w32}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x2)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
(0.9) \* (0.9991755753 \* (1 - 0.9991755753)) \* (2)

= 0.00009347692088

w32<sup>+</sup> = w32 - $\eta$ \* $\frac{\partial E}{\partial w32}$

= 0.6 - 0.5 \* 0.00009347692088

= 0.5999532615

$\frac{\partial E}{\partial w33}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x3)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
(0.9) \* (0.9991755753 \* (1 - 0.9991755753)) \* (3)

= 0.0001402153813

w33<sup>+</sup> = w33 - $\eta$ \* $\frac{\partial E}{\partial w23}$

= 0.65 - 0.5 \* 0.0001402153813

= 0.6499298923

$\frac{\partial E}{\partial w34}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x4)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
(0.9) \* (0.9991755753 \* (1 - 0.9991755753)) \* (3)

= 0.0001402153813

w34<sup>+</sup> = w34 - $\eta$ \* $\frac{\partial E}{\partial w34}$

= 0.7 - 0.5 \* 0.0001402153813

= 0.6999298923

<br>

`Langkah 9` :

$\frac{\partial E}{\partial biash1}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (1)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384)))
\* 0.8 \* (0.8320183851 \* (1 - 0.8320183851)) \* (1)

= 0.007048938982

biash1<sup>+</sup> = biash1 - $\eta$ \* $\frac{\partial E}{\partial biash1}$

= 0.25 - 0.5 \* 0.007048938982

= 0.2464755305

$\frac{\partial E}{\partial biash2}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (1)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
(0.85) \* (0.9872576505 \* (1 - 0.9872576505)) \* (1)

= 0.0006741212782

biash2<sup>+</sup> = biash2 - $\eta$ \* $\frac{\partial E}{\partial biash2}$

= 0.5 - 0.5 \* 0.0006741212782

= 0.4996629394

$\frac{\partial E}{\partial biash3}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (1)

= ((-2 \* (0 - 0.9662369384) \* (0.9662369384 \* (1 - 0.9662369384))) \*
(0.9) \* (0.9991755753 \* (1 - 0.9991755753)) \* (1)

= 0.00004673846044

biash3<sup>+</sup> = biash3 - $\eta$ \* $\frac{\partial E}{\partial biash3}$

= 0.75 - 0.5 \* 0.00004673846044

= 0.7499766308

 <br>

### Input 2

> x1 = 1; x2 = 2; x3 = 3; x4 = 3; Target = 0

#### Forward Pass

`Langkah 1` :

neth1 = w11 \* x1 + w12 \* x2 + w13 \* x3 + w14 \* x4 + biash1

= 0.04295106102 \* 1 + 0.09295106102 \* 2 + 0.1394265915 \* 3 +
0.1894265915 \* 3 + 0.2464755305

= 1.461888263

neth2 = w21 \* x1 + w22 \* x2 + w23 \* x3 + w24 \* x4 + biash2

= 0.2993258787 \* 1 + 0.3493258787 \* 2 + 0.3989888181 \* 3 +
0.4489888181 \* 3 + 0.4996629394

= 4.041573484

neth3 = w31 \* x1 + w32 \* x2 + w33 \* x3 + w34 \* x4 + biash3

= 0.5499532615 \* 1 + 0.5999532615 \* 2 + 0.6499298923 \* 3 +
0.6999298923 \* 3 + 0.7499766308

= 6.549415769

<br>

`Langkah 2` :

oh 1 = $\frac{1}{1 + e^{- neth1}}$

= $\frac{1}{1 + e^{- 1.461888263}}$

= 0.8118213098

oh 2 = $\frac{1}{1 + e^{- 4.041573484}}$

= $\frac{1}{1 + e^{- 4.35}}$

= 0.9827335631

oh 3 = $\frac{1}{1 + e^{- neth3}}$

= $\frac{1}{1 + e^{- 6.549415769}}$

= 0.9985710933

<br>

`Langkah 3` :

netout = wh1 \* oh1 + wh2 \* oh2 + wh3 \* oh3 + biasout

= 0.7737733985 \* 0.8118213098 + 0.8188799991 \* 0.9827335631 +
0.8685043263 \* 0.9985710933 + 0.918478339

= 3.218648247

<br>

`Langkah 4` :

out = $\frac{1}{1 + e^{- netout}}$

= $\frac{1}{1 + e^{- 3.218648247}}$

= 0.9615300443

<br>

`Langkah 5` :

E = (target - out)<sup>2</sup>

= (0 - 0.9615300443)<sup>2</sup>

= 0.9245400261

#### Backward Pass

`Langkah 6` :

$\frac{\partial E}{\partial wh1}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh1)

= (-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)) \*
(0.8118213098)

= 0.05774811953

wh1<sup>+</sup> = wh1 - $\eta$ \* $\frac{\partial E}{\partial wh1}$

= 0.7737733985 - 0.5 \* 0.05774811952

= 0.7448993388

$\frac{\partial E}{\partial wh2}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh2)

= (-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)) \*
(0.9827335631)

= 0.06990579649

wh2<sup>+</sup> = wh2 - $\eta$ \* $\frac{\partial E}{\partial wh2}$

= 0.8188799991 - 0.5 \* 0.06990579649

= 0.7839271008

$\frac{\partial E}{\partial wh3}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh3)

= (-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)) \*
(0.9985710933)

= 0.0710323838

wh3<sup>+</sup> = 0.8685043263 - $0.5$ \* 0.0710323838

= 0.8329881344

<br>

`Langkah 7` :

$\frac{\partial E}{\partial biasout}$ = (-2 \* (target - out)) \* (out
\* (1 - out)) \* (1)

= (-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)) \*
(1)

= 0.07113402769

biasout<sup>+</sup> = biasout - $\eta$ \* $\frac{\partial E}{\partial biasout}$

= 0.918478339 - 0.5 \* 0.07113402769

= 0.8829113252

<br>

`Langkah 8` :

$\frac{\partial E}{\partial w11}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x1)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.7737733985) \* (0.8118213098 \* (1 - 0.8118213098)) \* (1)

= 0.008408568823

w11<sup>+</sup> = w11 - $\eta$ \* $\frac{\partial E}{\partial w11}$

= 0.04295106102 - 0.5 \* 0.008408568823

= 0.03874677661

$\frac{\partial E}{\partial w12}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x2)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.7737733985) \* (0.8118213098 \* (1 - 0.8118213098)) \* (2)

= 0.01681713765

w12<sup>+</sup> = w12 - $\eta$ \* $\frac{\partial E}{\partial w12}$

= 0.09295106102 - 0.5 \* 0.01681713765

= 0.08454249219

$\frac{\partial E}{\partial w13}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x3)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.7737733985) \* (0.8118213098 \* (1 - 0.8118213098)) \* (3)

= 0.02522570647

w13<sup>+</sup> = w13 - $\eta$ \* $\frac{\partial E}{\partial w13}$

= 0.1394265915 - 0.5 \* 0.02522570647

= 0.1268137383

$\frac{\partial E}{\partial w14}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x4)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.7737733985) \* (0.8118213098 \* (1 - 0.8118213098)) \* (3)

= 0.02522570647

w14<sup>+</sup> = w14 - $\eta$ \* $\frac{\partial E}{\partial w14}$

= 0.1894265915 - 0.5 \* 0.02522570647

= 0.1768137383

$\frac{\partial E}{\partial w21}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x1)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8188799991) \* (0.9827335631 \* (1 - 0.9827335631)) \* (1)

= 0.0009884078318

w21<sup>+</sup> = w21 - $\eta$ \* $\frac{\partial E}{\partial 21}$

= 0.2993258787 - 0.5 \* 0.0009884078318

= 0.2988316748

$\frac{\partial E}{\partial w22}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x2)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8188799991) \* (0.9827335631 \* (1 - 0.9827335631)) \* (2)

= 0.001976815664

w22<sup>+</sup> = w22 - $\eta$ \* $\frac{\partial E}{\partial w22}$

= 0.3493258787 - 0.5 \* 0.001976815664

= 0.3483374709

$\frac{\partial E}{\partial w23}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x3)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8188799991) \* (0.9827335631 \* (1 - 0.9827335631)) \* (3)

= 0.002965223495

w23<sup>+</sup> = w23 - $\eta$ \* $\frac{\partial E}{\partial w23}$

= 0.3989888181 - 0.5 \* 0.002965223495

= 0.3975062063

$\frac{\partial E}{\partial w24}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x4)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* ( 0.8188799991) \* (0.9827335631 \* (1 - 0.9827335631)) \* (3)

= 0.002965223495

w24<sup>+</sup> = w24 - $\eta$ \* $\frac{\partial E}{\partial w24}$

= 0.4489888181 - 0.5 \* 0.002965223495

= 0.4475062063

$\frac{\partial E}{\partial w31}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x1)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8685043263) \* (0.9985710933\* (1 - 0.9985710933) \* (1)

= 0.00008815201589

w31<sup>+</sup> = w31 - $\eta$ \* $\frac{\partial E}{\partial 31}$

= 0.5499532615 - 0.5 \* 0.00008815201589

= 0.5499091855

$\frac{\partial E}{\partial w32}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x2)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8685043263) \* (0.9985710933\* (1 - 0.9985710933) \* (2)

= 0.0001763040318

w32<sup>+</sup> = w32 - $\eta$ \* $\frac{\partial E}{\partial w32}$

= 0.5999532615 - 0.5 \* 0.0001763040318

= 0.5998651095

$\frac{\partial E}{\partial w33}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x3)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8685043263) \* (0.9985710933\* (1 - 0.9985710933) \* (3)

= 0.0002644560477

w33<sup>+</sup> = w33 - $\eta$ \* $\frac{\partial E}{\partial w23}$

= 0.6499298923 - 0.5 \* 0.0002644560477

= 0.6497976643

$\frac{\partial E}{\partial w34}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x4)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8685043263) \* (0.9985710933\* (1 - 0.9985710933) \* (3)

= 0.0002644560477

w34<sup>+</sup> = w34 - $\eta$ \* $\frac{\partial E}{\partial w34}$

= 0.6999298923 - 0.5 \* 0.0002644560477

= 0.6997976643

<br>

`Langkah 9` :

$\frac{\partial E}{\partial biash1}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (1)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.7737733985) \* (0.8118213098 \* (1 - 0.8118213098)) \* (1)

= 0.008408568823

biash1<sup>+</sup> = biash1 - $\eta$ \* $\frac{\partial E}{\partial biash1}$

= 0.2464755305 - 0.5 \* 0.008408568823

= 0.2422712461

$\frac{\partial E}{\partial biash2}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (1)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8188799991) \* (0.9827335631 \* (1 - 0.9827335631)) \* (1)

= 0.0009884078318

biash2<sup>+</sup> = biash2 - $\eta$ \* $\frac{\partial E}{\partial biash2}$

= 0.4996629394 - 0.5 \* 0.0009884078318

= 0.4991687355

$\frac{\partial E}{\partial biash3}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (1)

= ((-2 \* (0 - 0.9615300443)) \* (0.9615300443 \* (1 - 0.9615300443)))
\* (0.8685043263) \* (0.9985710933\* (1 - 0.9985710933) \* (1)

= 0.00008815201589

biash3<sup>+</sup> = biash3 - $\eta$ \* $\frac{\partial E}{\partial biash3}$

= 0.7499766308 - 0.5 \* 0.00008815201589

= 0.7499325548

### Input 3

> x1 = 2; x2 = 1; x3 = 4; x4 = 1; Target = 1

#### Forward Pass

`Langkah 1` :
neth1 = w11 \* x1 + w12 \* x2 + w13 \* x3 + w14 \* x4 + biash1

= 0.03874677661 \* 2 + 0.08454249219 \* 1 + 0.1268137383 \* 4 +
0.1768137383 \* 1 + 0.2422712461

= 1.088375983

neth2 = w21 \* x1 + w22 \* x2 + w23 \* x3 + w24 \* x4 + biash2

= 0.2988316748 \* 2 + 0.3483374709 \* 1 + 0.3975062063 \* 4 +
0.4475062063 \* 1 + 0.4991687354

= 3.482700587

neth3 = w31 \* x1 + w32 \* x2 + w33 \* x3 + w34 \* x4 + biash3

= 0.5499091855 \* 2 + 0.5998651095 \* 1 + 0.6497976643 \* 4 +
0.6997976643 \* 1 + 0.7499325548

= 5.748604357

<br>

`Langkah 2` :

oh 1 = $\frac{1}{1 + e^{- neth1}}$

= $\frac{1}{1 + e^{- 1.088375983}}$

= 0.7480757853

oh 2 = $\frac{1}{1 + e^{- neth2}}$

= $\frac{1}{1 + e^{- 3.482700587}}$

= 0.9701915203

oh 3 = $\frac{1}{1 + e^{- neth3}}$

= $\frac{1}{1 + e^{- 5.748604357}}$

= 0.9968229002

<br>

`Langkah 3` :

netout = wh1 \* oh1 + wh2 \* oh2 + wh3 \* oh3 + biasout

= 0.7448993388 \* 0.7480757853 + 0.7839271008 \* 0.9701915203 +
0.8329881344 \* 0.9968229002 + 0.8829113252

= 3.031053557

<br>

`Langkah 4` :

out = $\frac{1}{1 + e^{- netout}}$

= $\frac{1}{1 + e^{- 3.031053557}}$

= 0.9539574701

<br>

`Langkah 5` :

E = (target - out)<sup>2</sup>

= (1 - 0.9539574701)<sup>2</sup>

= 0.002119914563

#### Backward Pass

`Langkah 6` :

$\frac{\partial E}{\partial wh1}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh1)

= (-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701)
\* (0.7480757853)

= -0.003025679784

wh1<sup>+</sup> = wh1 - $\eta$ \* $\frac{\partial E}{\partial wh1}$

= 0.7448993388 - 0.5 \* -0.003025679784

= 0.7464121787

$\frac{\partial E}{\partial wh2}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh2)

= (-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701)
\* (0.9701915203)

= -0.003924052787

wh2<sup>+</sup> = wh2 - $\eta$ \* $\frac{\partial E}{\partial wh2}$

= 0.7839271008 - 0.5 \* -0.003924052787

= 0.7858891272

$\frac{\partial E}{\partial wh3}$ = (-2 \* (target - out)) \* (out \* (1

- out)) \* (oh3)

= (-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701)
\* (0.9968229002)

= -0.004031766509

wh3<sup>+</sup> = wh3 - $\eta$ \* $\frac{\partial E}{\partial wh3}$

= 0.8329881344 - 0.5 \* -0.004031766509

= 0.8350040176

<br>

`Langkah 7` :

$\frac{\partial E}{\partial biasout}$ = (-2 \* (target - out)) \* (out
\* (1 - out)) \* (1)

= (-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701) \*
(1)

= -0.00404461666

biasout<sup>+</sup> = biasout - $\eta$ \* $\frac{\partial E}{\partial biasout}$

= 0.8829113252 - 0.5 \* -0.00404461666

= 0.8849336335

<br>

`Langkah 8` :

$\frac{\partial E}{\partial w11}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x1)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7448993388) \* (0.7480757853 \* (1 - 0.7480757853)) \* (2)

= -0.001135587129

w11<sup>+</sup> = w11 - $\eta$ \* $\frac{\partial E}{\partial w11}$

= 0.03874677661 - 0.5 \* -0.001135587129

= 0.03931457017

$\frac{\partial E}{\partial w12}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x2)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7448993388) \* (0.7480757853 \* (1 - 0.7480757853)) \* (1)

= -0.0005677935645

w12<sup>+</sup> = w12 - $\eta$ \* $\frac{\partial E}{\partial w12}$

= 0.08454249219 - 0.5 \* -0.0005677935645

= 0.08482638897

$\frac{\partial E}{\partial w13}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x3)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7448993388) \* (0.7480757853 \* (1 - 0.7480757853)) \* (4)

= -0.002271174258

w13<sup>+</sup> = w13 - $\eta$ \* $\frac{\partial E}{\partial w13}$

= 0.1268137383 - 0.5 \* -0.002271174258

= 0.1279493254

$\frac{\partial E}{\partial w14}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (x4)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7448993388) \* (0.7480757853 \* (1 - 0.7480757853)) \* (1)

= -0.0005677935645

w14<sup>+</sup> = w14 - $\eta$ \* $\frac{\partial E}{\partial w14}$

= 0.1768137383 - 0.5 \* -0.0005677935645

= 0.1770976351

$\frac{\partial E}{\partial w21}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x1)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7839271008) \* (0.9701915203 \* (1 - 0.9701915203)) \* (2)

= -0.000183391981

w21<sup>+</sup> = w21 - $\eta$ \* $\frac{\partial E}{\partial 21}$

= 0.2988316748 - 0.5 \* -0.000183391981

= 0.2989233708

$\frac{\partial E}{\partial w22}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x2)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7839271008) \* (0.9701915203 \* (1 - 0.9701915203)) \* (1)

= -0.00009169599048

w22<sup>+</sup> = w2 - $\eta$ \* $\frac{\partial E}{\partial w22}$

= 0.3483374709 - 0.5 \* -0.00009169599048

= 0.3483833189

$\frac{\partial E}{\partial w23}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x3)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7839271008) \* (0.9701915203 \* (1 - 0.9701915203)) \* (4)

= -0.0003667839619

w23<sup>+</sup> = w23 - $\eta$ \* $\frac{\partial E}{\partial w23}$

= 0.3975062063 - 0.5 \* -0.0003667839619

= 0.3976895983

$\frac{\partial E}{\partial w24}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (x4)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7839271008) \* (0.9701915203 \* (1 - 0.9701915203)) \* (1)

= -0.00009169599048

w24<sup>+</sup> = w24 - $\eta$ \* $\frac{\partial E}{\partial w24}$

= 0.4475062063 - 0.5 \* -0.00009169599048

= 0.4475520543

$\frac{\partial E}{\partial w31}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x1)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.8329881344) \* (0.9968229002\* (1 - 0.9968229002)) \* (2)

= -0.00002134003075

w31<sup>+</sup> = w31 - $\eta$ \* $\frac{\partial E}{\partial 31}$

= 0.5499091855 - 0.5 \* -0.00002134003075

= 0.5499198555

$\frac{\partial E}{\partial w32}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x2)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.8329881344) \* (0.9968229002\* (1 - 0.9968229002)) \* (1)

= -0.00001067001538

w32<sup>+</sup> = w32 - $\eta$ \* $\frac{\partial E}{\partial w32}$

= 0.5998651095 - 0.5 \* -0.00001067001538

= 0.5998704445

$\frac{\partial E}{\partial w33}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x3)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.8329881344) \* (0.9968229002\* (1 - 0.9968229002)) \* (4)

= -0.00004268006151

w33<sup>+</sup> = w33 - $\eta$ \* $\frac{\partial E}{\partial w23}$

= 0.6497976643 - 0.5 \* -0.00004268006151

= 0.6498190043

$\frac{\partial E}{\partial w34}$ = ((-2 \* (target - out)) \* (out \*
(1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (x4)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.8329881344) \* (0.9968229002\* (1 - 0.9968229002)) \* (1)

= -0.00001067001538

w34<sup>+</sup> = w34 - $\eta$ \* $\frac{\partial E}{\partial w34}$

= 0.6997976643 - 0.5 \* -0.00001067001538

= 0.6998029993

<br>

`Langkah 9` :

$\frac{\partial E}{\partial biash1}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh1) \* (oh1 \* (1 - oh1)) \* (1)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7448993388) \* (0.7480757853 \* (1 - 0.7480757853)) \* (1)

= -0.0005677935645

biash1<sup>+</sup> = biash1 - $\eta$ \* $\frac{\partial E}{\partial biash1}$

= 0.2422712461 - 0.5 \* -0.0005677935645

= 0.2425551429

$\frac{\partial E}{\partial biash2}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh2) \* (oh2 \* (1 - oh2)) \* (1)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.7839271008) \* (0.9701915203 \* (1 - 0.9701915203)) \* (1)

= -0.00009169599048

biash2<sup>+</sup> = biash2 - $\eta$ \* $\frac{\partial E}{\partial biash2}$

= 0.4991687354 - 0.5 \* -0.00009169599048

= 0.4992145834

$\frac{\partial E}{\partial biash3}$ = ((-2 \* (target - out)) \* (out
\* (1 - out))) \* (wh3) \* (oh3 \* (1 - oh3)) \* (1)

= ((-2 \* (1 - 0.9539574701)) \* ( 0.9539574701 \* (1 - 0.9539574701))
\* (0.8329881344) \* (0.9968229002\* (1 - 0.9968229002)) \* (1)

= -0.00001067001538

biash3<sup>+</sup> = biash3 - $\eta$ \* $\frac{\partial E}{\partial biash3}$

= 0.7499325548 - 0.5 \* -0.00001067001538

= 0.7499378898

<br>

## Result

### Excel

Hasil prediksi pada epoch ke-198, 199, 200 :

<br>

<div><img src="https://user-images.githubusercontent.com/107544829/190981681-305e4124-1662-48c7-91e9-fabfa864e636.png" width="600"/></div>
<div><img src="https://user-images.githubusercontent.com/107544829/190981733-0313f529-8639-403f-b02d-fd45cb3c6728.png" width="600"/></div>
<div><img src="https://user-images.githubusercontent.com/107544829/190981775-9c67607c-f0a5-47be-bf96-b9595a3e92a5.png" width="600"/></div>

<br>

### Python

<br>

<div><img src="https://user-images.githubusercontent.com/107544829/190982246-ad356762-ca06-4de7-93a9-48782fcbca19.png" width="600"/></div>
<div><img src="https://user-images.githubusercontent.com/107544829/190982267-295e2b25-c9a0-47ce-bf50-8e639ce8dcfa.png" width="600"/></div>
<div><img src="https://user-images.githubusercontent.com/107544829/190982288-fd442af7-c5bf-4b91-8be3-1a3970f306e9.png" width="600"/></div>

<br>

## Evaluation

Perhitungan menggunakan excel dan program python menggunakan langkah-langkah yang sama dan dapat memprediksi dengan akurasi 100% sesuai target pada epoch ke-200 dengan error root-mean-square error (RMSE) yang sangat minim, yaitu sebesar 0.000562221.

<br>

Referensi :

- [Machine Learning : Single Layer Perceptron](https://vincentmichael089.medium.com/machine-learning-1-single-layer-perceptron-9d94c62f1970)
- [A Step by Step Backpropagation Example by Matt Mazur](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
- [Machine Learning 101 by Ridwan Ilyas](https://www.youtube.com/playlist?list=PLo6nZTcpSz2p5oKKkg6ZWHx4Pw7ToYVtD)

