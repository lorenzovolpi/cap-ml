# rcv1_GCAT_9prevs

## 10% positives
> train: [0.90005165 0.09994835]  
> validation: [0.90005165 0.09994835]  
> bin_sld: 238.632s  
> mul_sld: 101.343s  
> mul_sld_gs: 441.747s  
> bin_cc: 230.303s  
> mul_cc: 77.603s  
> kfcv: 66.959s  
> ref: 65.305s  
> atc_mc: 66.465s  
> atc_ne: 65.103s  
> doc_feat: 56.750s  
> tot: 446.654s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0062</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.3429</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.3260</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.3360</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.3402</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.3351</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.3371</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.3313</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.3378</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.3337</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.3346</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.3327</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.3406</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.3326</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.3335</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.3359</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.3324</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.3378</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.3373</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.3323</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.3353</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.3196</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_10_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_10_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_10_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_10_f1.png)


## 20% positives
> train: [0.79984504 0.20015496]  
> validation: [0.80010331 0.19989669]  
> bin_sld: 431.443s  
> mul_sld: 188.845s  
> bin_sld_gs: 1093.467s  
> mul_sld_gs: 642.106s  
> bin_sld_gsq: 522.073s  
> bin_pacc: 422.062s  
> mul_pacc: 154.976s  
> binmc_pacc: 417.137s  
> mulmc_pacc: 140.524s  
> binne_pacc: 421.581s  
> mulne_pacc: 157.883s  
> bin_pacc_gs: 723.261s  
> mul_pacc_gs: 258.223s  
> bin_cc: 402.467s  
> mul_cc: 144.805s  
> kfcv: 108.333s  
> ref: 118.158s  
> atc_mc: 118.502s  
> atc_ne: 96.965s  
> doc_feat: 91.141s  
> tot: 1096.718s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.2307</td>
      <td>0.1390</td>
      <td>0.0183</td>
      <td>0.0000</td>
      <td>0.1580</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0818</td>
      <td>0.1435</td>
      <td>0.1385</td>
      <td>0.4540</td>
      <td>0.1115</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0575</td>
      <td>0.1036</td>
      <td>0.1008</td>
      <td>0.3316</td>
      <td>0.0899</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0505</td>
      <td>0.0911</td>
      <td>0.0806</td>
      <td>0.2445</td>
      <td>0.0782</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0449</td>
      <td>0.0755</td>
      <td>0.0703</td>
      <td>0.2092</td>
      <td>0.0673</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0408</td>
      <td>0.0700</td>
      <td>0.0626</td>
      <td>0.1771</td>
      <td>0.0637</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0362</td>
      <td>0.0637</td>
      <td>0.0532</td>
      <td>0.1534</td>
      <td>0.0612</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0348</td>
      <td>0.0638</td>
      <td>0.0519</td>
      <td>0.1498</td>
      <td>0.0553</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0356</td>
      <td>0.0655</td>
      <td>0.0559</td>
      <td>0.1511</td>
      <td>0.0605</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0354</td>
      <td>0.0638</td>
      <td>0.0541</td>
      <td>0.1399</td>
      <td>0.0589</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0351</td>
      <td>0.0626</td>
      <td>0.0527</td>
      <td>0.1404</td>
      <td>0.0587</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0312</td>
      <td>0.0580</td>
      <td>0.0473</td>
      <td>0.1301</td>
      <td>0.0550</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0322</td>
      <td>0.0563</td>
      <td>0.0430</td>
      <td>0.1160</td>
      <td>0.0526</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0320</td>
      <td>0.0574</td>
      <td>0.0441</td>
      <td>0.1223</td>
      <td>0.0541</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0291</td>
      <td>0.0543</td>
      <td>0.0415</td>
      <td>0.1163</td>
      <td>0.0511</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0291</td>
      <td>0.0532</td>
      <td>0.0419</td>
      <td>0.1111</td>
      <td>0.0521</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0293</td>
      <td>0.0546</td>
      <td>0.0410</td>
      <td>0.1101</td>
      <td>0.0526</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0262</td>
      <td>0.0495</td>
      <td>0.0348</td>
      <td>0.0982</td>
      <td>0.0488</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0282</td>
      <td>0.0507</td>
      <td>0.0391</td>
      <td>0.1046</td>
      <td>0.0503</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0291</td>
      <td>0.0528</td>
      <td>0.0410</td>
      <td>0.1089</td>
      <td>0.0518</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0239</td>
      <td>0.0523</td>
      <td>0.0327</td>
      <td>0.1016</td>
      <td>0.0495</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0464</td>
      <td>0.0705</td>
      <td>0.0545</td>
      <td>0.1557</td>
      <td>0.0658</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_20_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_20_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_20_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_20_f1.png)


## 30% positives
> train: [0.69989669 0.30010331]  
> validation: [0.70015496 0.29984504]  
> bin_sld: 440.439s  
> mul_sld: 181.173s  
> bin_sld_gs: 1117.061s  
> mul_sld_gs: 644.511s  
> bin_sld_gsq: 529.624s  
> bin_pacc: 425.791s  
> mul_pacc: 174.577s  
> binmc_pacc: 425.760s  
> mulmc_pacc: 147.208s  
> binne_pacc: 424.010s  
> mulne_pacc: 137.064s  
> bin_pacc_gs: 729.394s  
> mul_pacc_gs: 256.419s  
> bin_cc: 407.241s  
> mul_cc: 148.864s  
> kfcv: 129.860s  
> ref: 89.336s  
> atc_mc: 117.685s  
> atc_ne: 123.998s  
> doc_feat: 91.912s  
> tot: 1120.968s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.4489</td>
      <td>0.0707</td>
      <td>0.3039</td>
      <td>0.1143</td>
      <td>0.3913</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0570</td>
      <td>0.0612</td>
      <td>0.0556</td>
      <td>0.0747</td>
      <td>0.0484</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0428</td>
      <td>0.0414</td>
      <td>0.0378</td>
      <td>0.0479</td>
      <td>0.0293</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0254</td>
      <td>0.0325</td>
      <td>0.0376</td>
      <td>0.0491</td>
      <td>0.0291</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0217</td>
      <td>0.0247</td>
      <td>0.0298</td>
      <td>0.0399</td>
      <td>0.0239</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0193</td>
      <td>0.0217</td>
      <td>0.0248</td>
      <td>0.0359</td>
      <td>0.0193</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0163</td>
      <td>0.0176</td>
      <td>0.0237</td>
      <td>0.0352</td>
      <td>0.0189</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0145</td>
      <td>0.0155</td>
      <td>0.0216</td>
      <td>0.0316</td>
      <td>0.0161</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0161</td>
      <td>0.0144</td>
      <td>0.0220</td>
      <td>0.0339</td>
      <td>0.0145</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0155</td>
      <td>0.0139</td>
      <td>0.0205</td>
      <td>0.0325</td>
      <td>0.0131</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0140</td>
      <td>0.0116</td>
      <td>0.0180</td>
      <td>0.0294</td>
      <td>0.0110</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0136</td>
      <td>0.0121</td>
      <td>0.0174</td>
      <td>0.0298</td>
      <td>0.0115</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0148</td>
      <td>0.0106</td>
      <td>0.0188</td>
      <td>0.0323</td>
      <td>0.0109</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0152</td>
      <td>0.0106</td>
      <td>0.0184</td>
      <td>0.0317</td>
      <td>0.0113</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0141</td>
      <td>0.0091</td>
      <td>0.0172</td>
      <td>0.0306</td>
      <td>0.0088</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0141</td>
      <td>0.0089</td>
      <td>0.0159</td>
      <td>0.0286</td>
      <td>0.0093</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0144</td>
      <td>0.0081</td>
      <td>0.0163</td>
      <td>0.0299</td>
      <td>0.0089</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0142</td>
      <td>0.0084</td>
      <td>0.0159</td>
      <td>0.0284</td>
      <td>0.0087</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0137</td>
      <td>0.0076</td>
      <td>0.0150</td>
      <td>0.0277</td>
      <td>0.0083</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0159</td>
      <td>0.0090</td>
      <td>0.0172</td>
      <td>0.0307</td>
      <td>0.0086</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0125</td>
      <td>0.0080</td>
      <td>0.0129</td>
      <td>0.0284</td>
      <td>0.0073</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0397</td>
      <td>0.0199</td>
      <td>0.0362</td>
      <td>0.0392</td>
      <td>0.0337</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_30_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_30_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_30_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_30_f1.png)


## 40% positives
> train: [0.59994835 0.40005165]  
> validation: [0.59994835 0.40005165]  
> bin_sld: 441.902s  
> mul_sld: 171.147s  
> bin_sld_gs: 1129.143s  
> mul_sld_gs: 661.698s  
> bin_sld_gsq: 534.997s  
> bin_pacc: 438.840s  
> mul_pacc: 172.945s  
> binmc_pacc: 434.527s  
> mulmc_pacc: 166.147s  
> binne_pacc: 431.135s  
> mulne_pacc: 144.562s  
> bin_pacc_gs: 735.193s  
> mul_pacc_gs: 266.608s  
> bin_cc: 412.883s  
> mul_cc: 159.067s  
> kfcv: 137.954s  
> ref: 132.249s  
> atc_mc: 137.719s  
> atc_ne: 134.107s  
> doc_feat: 79.638s  
> tot: 1133.285s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.2187</td>
      <td>0.1510</td>
      <td>0.1220</td>
      <td>0.1634</td>
      <td>0.5122</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0528</td>
      <td>0.0637</td>
      <td>0.0498</td>
      <td>0.0520</td>
      <td>0.0575</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0386</td>
      <td>0.0449</td>
      <td>0.0367</td>
      <td>0.0378</td>
      <td>0.0264</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0262</td>
      <td>0.0302</td>
      <td>0.0282</td>
      <td>0.0257</td>
      <td>0.0185</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0216</td>
      <td>0.0263</td>
      <td>0.0240</td>
      <td>0.0222</td>
      <td>0.0141</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0192</td>
      <td>0.0222</td>
      <td>0.0212</td>
      <td>0.0198</td>
      <td>0.0145</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0162</td>
      <td>0.0172</td>
      <td>0.0181</td>
      <td>0.0176</td>
      <td>0.0119</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0141</td>
      <td>0.0159</td>
      <td>0.0151</td>
      <td>0.0151</td>
      <td>0.0098</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0143</td>
      <td>0.0133</td>
      <td>0.0134</td>
      <td>0.0138</td>
      <td>0.0093</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0140</td>
      <td>0.0132</td>
      <td>0.0137</td>
      <td>0.0138</td>
      <td>0.0074</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0148</td>
      <td>0.0128</td>
      <td>0.0130</td>
      <td>0.0138</td>
      <td>0.0071</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0159</td>
      <td>0.0118</td>
      <td>0.0139</td>
      <td>0.0141</td>
      <td>0.0076</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0173</td>
      <td>0.0116</td>
      <td>0.0143</td>
      <td>0.0144</td>
      <td>0.0067</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0176</td>
      <td>0.0106</td>
      <td>0.0141</td>
      <td>0.0140</td>
      <td>0.0059</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0195</td>
      <td>0.0105</td>
      <td>0.0152</td>
      <td>0.0154</td>
      <td>0.0057</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0190</td>
      <td>0.0103</td>
      <td>0.0149</td>
      <td>0.0147</td>
      <td>0.0063</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0192</td>
      <td>0.0095</td>
      <td>0.0147</td>
      <td>0.0141</td>
      <td>0.0053</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0214</td>
      <td>0.0114</td>
      <td>0.0175</td>
      <td>0.0166</td>
      <td>0.0055</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0215</td>
      <td>0.0105</td>
      <td>0.0175</td>
      <td>0.0164</td>
      <td>0.0050</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0227</td>
      <td>0.0117</td>
      <td>0.0180</td>
      <td>0.0168</td>
      <td>0.0049</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0214</td>
      <td>0.0119</td>
      <td>0.0141</td>
      <td>0.0170</td>
      <td>0.0054</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0308</td>
      <td>0.0248</td>
      <td>0.0243</td>
      <td>0.0261</td>
      <td>0.0356</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_40_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_40_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_40_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_40_f1.png)


## 50% positives
> train: [0.5 0.5]  
> validation: [0.5 0.5]  
> bin_sld: 432.840s  
> mul_sld: 159.373s  
> bin_sld_gs: 1148.243s  
> mul_sld_gs: 657.176s  
> bin_sld_gsq: 539.408s  
> bin_pacc: 430.121s  
> mul_pacc: 173.440s  
> binmc_pacc: 433.871s  
> mulmc_pacc: 158.794s  
> binne_pacc: 435.732s  
> mulne_pacc: 158.199s  
> bin_pacc_gs: 728.918s  
> mul_pacc_gs: 263.162s  
> bin_cc: 405.769s  
> mul_cc: 134.145s  
> kfcv: 125.438s  
> ref: 127.690s  
> atc_mc: 108.387s  
> atc_ne: 115.014s  
> doc_feat: 90.803s  
> tot: 1152.446s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0267</td>
      <td>0.4403</td>
      <td>0.3449</td>
      <td>0.1107</td>
      <td>0.5861</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0606</td>
      <td>0.1552</td>
      <td>0.1050</td>
      <td>0.0528</td>
      <td>0.1441</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0311</td>
      <td>0.0903</td>
      <td>0.0636</td>
      <td>0.0371</td>
      <td>0.0650</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0270</td>
      <td>0.0630</td>
      <td>0.0470</td>
      <td>0.0273</td>
      <td>0.0438</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0230</td>
      <td>0.0438</td>
      <td>0.0329</td>
      <td>0.0225</td>
      <td>0.0258</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0202</td>
      <td>0.0365</td>
      <td>0.0290</td>
      <td>0.0206</td>
      <td>0.0206</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0172</td>
      <td>0.0276</td>
      <td>0.0221</td>
      <td>0.0147</td>
      <td>0.0140</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0152</td>
      <td>0.0251</td>
      <td>0.0194</td>
      <td>0.0150</td>
      <td>0.0145</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0157</td>
      <td>0.0191</td>
      <td>0.0151</td>
      <td>0.0137</td>
      <td>0.0118</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0127</td>
      <td>0.0173</td>
      <td>0.0131</td>
      <td>0.0123</td>
      <td>0.0089</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0131</td>
      <td>0.0133</td>
      <td>0.0108</td>
      <td>0.0115</td>
      <td>0.0096</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0134</td>
      <td>0.0116</td>
      <td>0.0103</td>
      <td>0.0116</td>
      <td>0.0084</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0121</td>
      <td>0.0098</td>
      <td>0.0089</td>
      <td>0.0108</td>
      <td>0.0079</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0119</td>
      <td>0.0078</td>
      <td>0.0086</td>
      <td>0.0113</td>
      <td>0.0072</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0124</td>
      <td>0.0073</td>
      <td>0.0094</td>
      <td>0.0120</td>
      <td>0.0060</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0106</td>
      <td>0.0070</td>
      <td>0.0088</td>
      <td>0.0113</td>
      <td>0.0064</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0097</td>
      <td>0.0063</td>
      <td>0.0088</td>
      <td>0.0110</td>
      <td>0.0062</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0114</td>
      <td>0.0077</td>
      <td>0.0113</td>
      <td>0.0135</td>
      <td>0.0051</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0101</td>
      <td>0.0069</td>
      <td>0.0111</td>
      <td>0.0131</td>
      <td>0.0048</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0101</td>
      <td>0.0069</td>
      <td>0.0115</td>
      <td>0.0130</td>
      <td>0.0051</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0067</td>
      <td>0.0079</td>
      <td>0.0098</td>
      <td>0.0140</td>
      <td>0.0047</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0177</td>
      <td>0.0481</td>
      <td>0.0382</td>
      <td>0.0219</td>
      <td>0.0479</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_50_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_50_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_50_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_50_f1.png)


## 60% positives
> train: [0.40005165 0.59994835]  
> validation: [0.40005165 0.59994835]  
> bin_sld: 423.693s  
> mul_sld: 164.785s  
> bin_sld_gs: 1105.830s  
> mul_sld_gs: 648.000s  
> bin_sld_gsq: 523.679s  
> bin_pacc: 415.735s  
> mul_pacc: 142.337s  
> binmc_pacc: 422.073s  
> mulmc_pacc: 162.522s  
> binne_pacc: 411.703s  
> mulne_pacc: 159.753s  
> bin_pacc_gs: 722.307s  
> mul_pacc_gs: 263.296s  
> bin_cc: 394.807s  
> mul_cc: 127.886s  
> kfcv: 118.430s  
> ref: 114.366s  
> atc_mc: 111.695s  
> atc_ne: 96.170s  
> doc_feat: 92.012s  
> tot: 1110.535s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.1390</td>
      <td>0.0057</td>
      <td>0.2345</td>
      <td>0.0501</td>
      <td>0.6121</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0698</td>
      <td>0.0825</td>
      <td>0.1005</td>
      <td>0.0415</td>
      <td>0.2334</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0437</td>
      <td>0.0508</td>
      <td>0.0536</td>
      <td>0.0320</td>
      <td>0.1135</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0361</td>
      <td>0.0339</td>
      <td>0.0432</td>
      <td>0.0216</td>
      <td>0.0789</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0254</td>
      <td>0.0291</td>
      <td>0.0274</td>
      <td>0.0195</td>
      <td>0.0503</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0242</td>
      <td>0.0216</td>
      <td>0.0243</td>
      <td>0.0161</td>
      <td>0.0383</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0200</td>
      <td>0.0183</td>
      <td>0.0196</td>
      <td>0.0114</td>
      <td>0.0278</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0189</td>
      <td>0.0143</td>
      <td>0.0148</td>
      <td>0.0117</td>
      <td>0.0211</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0187</td>
      <td>0.0113</td>
      <td>0.0134</td>
      <td>0.0102</td>
      <td>0.0176</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0162</td>
      <td>0.0120</td>
      <td>0.0116</td>
      <td>0.0093</td>
      <td>0.0130</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0158</td>
      <td>0.0102</td>
      <td>0.0102</td>
      <td>0.0088</td>
      <td>0.0118</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0147</td>
      <td>0.0105</td>
      <td>0.0101</td>
      <td>0.0090</td>
      <td>0.0098</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0131</td>
      <td>0.0098</td>
      <td>0.0080</td>
      <td>0.0075</td>
      <td>0.0088</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0112</td>
      <td>0.0093</td>
      <td>0.0073</td>
      <td>0.0070</td>
      <td>0.0069</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0089</td>
      <td>0.0080</td>
      <td>0.0059</td>
      <td>0.0060</td>
      <td>0.0057</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0083</td>
      <td>0.0088</td>
      <td>0.0057</td>
      <td>0.0059</td>
      <td>0.0052</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0071</td>
      <td>0.0080</td>
      <td>0.0050</td>
      <td>0.0054</td>
      <td>0.0046</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0059</td>
      <td>0.0072</td>
      <td>0.0049</td>
      <td>0.0053</td>
      <td>0.0039</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0055</td>
      <td>0.0075</td>
      <td>0.0050</td>
      <td>0.0052</td>
      <td>0.0038</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0042</td>
      <td>0.0064</td>
      <td>0.0048</td>
      <td>0.0048</td>
      <td>0.0027</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0032</td>
      <td>0.0049</td>
      <td>0.0044</td>
      <td>0.0044</td>
      <td>0.0026</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0243</td>
      <td>0.0176</td>
      <td>0.0293</td>
      <td>0.0139</td>
      <td>0.0606</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_60_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_60_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_60_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_60_f1.png)


## 70% positives
> train: [0.29984504 0.70015496]  
> validation: [0.30010331 0.69989669]  
> bin_sld: 446.445s  
> mul_sld: 164.730s  
> bin_sld_gs: 1117.398s  
> mul_sld_gs: 661.430s  
> bin_sld_gsq: 528.925s  
> bin_pacc: 431.969s  
> mul_pacc: 151.272s  
> binmc_pacc: 430.157s  
> mulmc_pacc: 180.251s  
> binne_pacc: 438.442s  
> mulne_pacc: 173.700s  
> bin_pacc_gs: 733.397s  
> mul_pacc_gs: 262.705s  
> bin_cc: 407.885s  
> mul_cc: 153.959s  
> kfcv: 132.641s  
> ref: 136.486s  
> atc_mc: 135.594s  
> atc_ne: 129.372s  
> doc_feat: 95.722s  
> tot: 1121.806s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0873</td>
      <td>0.0886</td>
      <td>0.1223</td>
      <td>0.1188</td>
      <td>0.5530</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0497</td>
      <td>0.0496</td>
      <td>0.0702</td>
      <td>0.0690</td>
      <td>0.2717</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0319</td>
      <td>0.0320</td>
      <td>0.0426</td>
      <td>0.0409</td>
      <td>0.1554</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0245</td>
      <td>0.0254</td>
      <td>0.0320</td>
      <td>0.0312</td>
      <td>0.0993</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0155</td>
      <td>0.0191</td>
      <td>0.0208</td>
      <td>0.0190</td>
      <td>0.0643</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0128</td>
      <td>0.0159</td>
      <td>0.0175</td>
      <td>0.0153</td>
      <td>0.0433</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0115</td>
      <td>0.0115</td>
      <td>0.0167</td>
      <td>0.0131</td>
      <td>0.0320</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0099</td>
      <td>0.0143</td>
      <td>0.0117</td>
      <td>0.0101</td>
      <td>0.0194</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0076</td>
      <td>0.0111</td>
      <td>0.0098</td>
      <td>0.0082</td>
      <td>0.0151</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0081</td>
      <td>0.0130</td>
      <td>0.0085</td>
      <td>0.0082</td>
      <td>0.0116</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0076</td>
      <td>0.0104</td>
      <td>0.0095</td>
      <td>0.0071</td>
      <td>0.0101</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0084</td>
      <td>0.0129</td>
      <td>0.0073</td>
      <td>0.0073</td>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0073</td>
      <td>0.0115</td>
      <td>0.0072</td>
      <td>0.0066</td>
      <td>0.0054</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0072</td>
      <td>0.0111</td>
      <td>0.0065</td>
      <td>0.0060</td>
      <td>0.0054</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0082</td>
      <td>0.0127</td>
      <td>0.0060</td>
      <td>0.0063</td>
      <td>0.0044</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0082</td>
      <td>0.0129</td>
      <td>0.0057</td>
      <td>0.0064</td>
      <td>0.0042</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0094</td>
      <td>0.0136</td>
      <td>0.0061</td>
      <td>0.0073</td>
      <td>0.0051</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0099</td>
      <td>0.0137</td>
      <td>0.0065</td>
      <td>0.0077</td>
      <td>0.0046</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0098</td>
      <td>0.0133</td>
      <td>0.0059</td>
      <td>0.0077</td>
      <td>0.0045</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0110</td>
      <td>0.0142</td>
      <td>0.0072</td>
      <td>0.0086</td>
      <td>0.0044</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0122</td>
      <td>0.0149</td>
      <td>0.0067</td>
      <td>0.0099</td>
      <td>0.0051</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0171</td>
      <td>0.0201</td>
      <td>0.0203</td>
      <td>0.0197</td>
      <td>0.0631</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_70_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_70_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_70_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_70_f1.png)


## 80% positives
> train: [0.19989669 0.80010331]  
> validation: [0.20015496 0.79984504]  
> bin_sld: 407.545s  
> mul_sld: 147.073s  
> bin_sld_gs: 4046.450s  
> mul_sld_gs: 3606.019s  
> bin_sld_gsq: 490.593s  
> bin_pacc: 399.480s  
> mul_pacc: 162.231s  
> binmc_pacc: 403.466s  
> mulmc_pacc: 179.691s  
> binne_pacc: 403.638s  
> mulne_pacc: 171.078s  
> bin_pacc_gs: 3446.185s  
> mul_pacc_gs: 257.968s  
> bin_cc: 374.905s  
> mul_cc: 150.379s  
> kfcv: 133.439s  
> ref: 132.673s  
> atc_mc: 130.594s  
> atc_ne: 131.031s  
> doc_feat: 81.112s  
> tot: 4051.202s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0014</td>
      <td>0.0411</td>
      <td>0.0663</td>
      <td>0.0200</td>
      <td>0.5543</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0893</td>
      <td>0.0360</td>
      <td>0.0397</td>
      <td>0.0492</td>
      <td>0.3560</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0607</td>
      <td>0.0242</td>
      <td>0.0324</td>
      <td>0.0324</td>
      <td>0.2466</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0407</td>
      <td>0.0231</td>
      <td>0.0237</td>
      <td>0.0282</td>
      <td>0.1736</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0295</td>
      <td>0.0187</td>
      <td>0.0199</td>
      <td>0.0223</td>
      <td>0.1294</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0255</td>
      <td>0.0155</td>
      <td>0.0153</td>
      <td>0.0186</td>
      <td>0.0960</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0169</td>
      <td>0.0121</td>
      <td>0.0134</td>
      <td>0.0135</td>
      <td>0.0688</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0162</td>
      <td>0.0137</td>
      <td>0.0125</td>
      <td>0.0144</td>
      <td>0.0486</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0110</td>
      <td>0.0112</td>
      <td>0.0094</td>
      <td>0.0108</td>
      <td>0.0379</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0116</td>
      <td>0.0108</td>
      <td>0.0108</td>
      <td>0.0116</td>
      <td>0.0268</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0076</td>
      <td>0.0080</td>
      <td>0.0079</td>
      <td>0.0080</td>
      <td>0.0223</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0069</td>
      <td>0.0087</td>
      <td>0.0082</td>
      <td>0.0080</td>
      <td>0.0150</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0057</td>
      <td>0.0068</td>
      <td>0.0067</td>
      <td>0.0062</td>
      <td>0.0119</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0056</td>
      <td>0.0064</td>
      <td>0.0065</td>
      <td>0.0066</td>
      <td>0.0086</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0052</td>
      <td>0.0059</td>
      <td>0.0071</td>
      <td>0.0058</td>
      <td>0.0067</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0044</td>
      <td>0.0049</td>
      <td>0.0062</td>
      <td>0.0046</td>
      <td>0.0046</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0050</td>
      <td>0.0044</td>
      <td>0.0067</td>
      <td>0.0046</td>
      <td>0.0044</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0050</td>
      <td>0.0040</td>
      <td>0.0060</td>
      <td>0.0040</td>
      <td>0.0036</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0062</td>
      <td>0.0035</td>
      <td>0.0049</td>
      <td>0.0039</td>
      <td>0.0034</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0055</td>
      <td>0.0035</td>
      <td>0.0054</td>
      <td>0.0038</td>
      <td>0.0036</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0021</td>
      <td>0.0025</td>
      <td>0.0049</td>
      <td>0.0022</td>
      <td>0.0041</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0172</td>
      <td>0.0126</td>
      <td>0.0149</td>
      <td>0.0133</td>
      <td>0.0870</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_80_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_80_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_80_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_80_f1.png)


## 90% positives
> train: [0.09994835 0.90005165]  
> validation: [0.09994835 0.90005165]  
> mul_sld: 56.896s  
> mul_sld_gs: 351.569s  
> mul_cc: 51.884s  
> kfcv: 46.187s  
> ref: 44.331s  
> atc_mc: 47.954s  
> atc_ne: 47.496s  
> doc_feat: 41.155s  
> tot: 357.732s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.4608</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.3731</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.3053</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.2498</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.2053</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.1687</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.1354</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.1093</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0883</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0693</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0554</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0420</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0326</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0245</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0169</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0103</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0063</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0038</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0032</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0038</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0045</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.1128</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_90_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_90_f1.png)
![plot_diagonal](plot/diagonal_rcv1_GCAT_max_conf_vs_atc_pacc_90_f1.png)
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_90_f1.png)


## avg
### avg on train
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.1647</td>
      <td>0.1338</td>
      <td>0.1732</td>
      <td>0.0825</td>
      <td>0.4260</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0659</td>
      <td>0.0845</td>
      <td>0.0799</td>
      <td>0.1133</td>
      <td>0.2154</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0438</td>
      <td>0.0553</td>
      <td>0.0525</td>
      <td>0.0799</td>
      <td>0.1508</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0329</td>
      <td>0.0427</td>
      <td>0.0418</td>
      <td>0.0611</td>
      <td>0.1230</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0260</td>
      <td>0.0339</td>
      <td>0.0322</td>
      <td>0.0506</td>
      <td>0.1023</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0231</td>
      <td>0.0291</td>
      <td>0.0278</td>
      <td>0.0433</td>
      <td>0.0888</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0192</td>
      <td>0.0240</td>
      <td>0.0238</td>
      <td>0.0370</td>
      <td>0.0786</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0177</td>
      <td>0.0232</td>
      <td>0.0210</td>
      <td>0.0354</td>
      <td>0.0695</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0170</td>
      <td>0.0208</td>
      <td>0.0199</td>
      <td>0.0345</td>
      <td>0.0659</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0162</td>
      <td>0.0206</td>
      <td>0.0189</td>
      <td>0.0325</td>
      <td>0.0603</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0154</td>
      <td>0.0184</td>
      <td>0.0174</td>
      <td>0.0313</td>
      <td>0.0578</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0149</td>
      <td>0.0179</td>
      <td>0.0164</td>
      <td>0.0300</td>
      <td>0.0542</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0146</td>
      <td>0.0166</td>
      <td>0.0153</td>
      <td>0.0277</td>
      <td>0.0530</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0144</td>
      <td>0.0162</td>
      <td>0.0150</td>
      <td>0.0284</td>
      <td>0.0507</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0139</td>
      <td>0.0154</td>
      <td>0.0146</td>
      <td>0.0275</td>
      <td>0.0488</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0134</td>
      <td>0.0152</td>
      <td>0.0142</td>
      <td>0.0261</td>
      <td>0.0483</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0135</td>
      <td>0.0149</td>
      <td>0.0141</td>
      <td>0.0261</td>
      <td>0.0473</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0134</td>
      <td>0.0146</td>
      <td>0.0138</td>
      <td>0.0248</td>
      <td>0.0469</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0136</td>
      <td>0.0143</td>
      <td>0.0141</td>
      <td>0.0255</td>
      <td>0.0467</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0141</td>
      <td>0.0149</td>
      <td>0.0150</td>
      <td>0.0267</td>
      <td>0.0464</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0117</td>
      <td>0.0146</td>
      <td>0.0122</td>
      <td>0.0254</td>
      <td>0.0465</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0276</td>
      <td>0.0305</td>
      <td>0.0311</td>
      <td>0.0414</td>
      <td>0.0918</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_avg_train_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_avg_train_f1.png)
### avg on test
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_pacc</th>
      <th>mul_pacc</th>
      <th>binmc_pacc</th>
      <th>mulmc_pacc</th>
      <th>atc_mc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.3196</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0464</td>
      <td>0.0705</td>
      <td>0.0545</td>
      <td>0.1557</td>
      <td>0.0658</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0397</td>
      <td>0.0199</td>
      <td>0.0362</td>
      <td>0.0392</td>
      <td>0.0337</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0308</td>
      <td>0.0248</td>
      <td>0.0243</td>
      <td>0.0261</td>
      <td>0.0356</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0177</td>
      <td>0.0481</td>
      <td>0.0382</td>
      <td>0.0219</td>
      <td>0.0479</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0243</td>
      <td>0.0176</td>
      <td>0.0293</td>
      <td>0.0139</td>
      <td>0.0606</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0171</td>
      <td>0.0201</td>
      <td>0.0203</td>
      <td>0.0197</td>
      <td>0.0631</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0172</td>
      <td>0.0126</td>
      <td>0.0149</td>
      <td>0.0133</td>
      <td>0.0870</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1128</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0276</td>
      <td>0.0305</td>
      <td>0.0311</td>
      <td>0.0414</td>
      <td>0.0918</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_rcv1_GCAT_max_conf_vs_atc_pacc_avg_test_f1.png)
![plot_delta_stdev](plot/delta_stdev_rcv1_GCAT_max_conf_vs_atc_pacc_avg_test_f1.png)
### avg dataset shift
![plot_shift](plot/shift_rcv1_GCAT_max_conf_vs_atc_pacc_avg_f1.png)
