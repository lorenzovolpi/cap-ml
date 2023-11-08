# imdb_9prevs

## 10% positives
> train: [0.89991359 0.10008641]  
> validation: [0.9000576 0.0999424]  
> bin_sld: 249.048s  
> mul_sld: 71.459s  
> mul_sld_gs: 455.993s  
> bin_pacc: 244.563s  
> mul_pacc: 43.016s  
> binmc_pacc: 240.469s  
> mulmc_pacc: 41.750s  
> binne_pacc: 236.217s  
> mulne_pacc: 34.041s  
> bin_cc: 241.914s  
> mul_cc: 40.328s  
> kfcv: 24.723s  
> ref: 18.562s  
> atc_mc: 27.491s  
> atc_ne: 25.568s  
> doc_feat: 11.526s  
> tot: 456.775s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mul_sld_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0220</td>
      <td>0.0993</td>
      <td>0.0370</td>
      <td>0.0977</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0151</td>
      <td>0.0499</td>
      <td>0.0155</td>
      <td>0.0484</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0114</td>
      <td>0.0009</td>
      <td>0.0067</td>
      <td>0.0013</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0072</td>
      <td>0.0490</td>
      <td>0.0255</td>
      <td>0.0503</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0088</td>
      <td>0.0982</td>
      <td>0.0457</td>
      <td>0.0995</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0145</td>
      <td>0.1476</td>
      <td>0.0664</td>
      <td>0.1488</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0199</td>
      <td>0.1966</td>
      <td>0.0858</td>
      <td>0.1978</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0276</td>
      <td>0.2463</td>
      <td>0.1100</td>
      <td>0.2474</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0348</td>
      <td>0.2960</td>
      <td>0.1302</td>
      <td>0.2970</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0394</td>
      <td>0.3448</td>
      <td>0.1477</td>
      <td>0.3457</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0455</td>
      <td>0.3940</td>
      <td>0.1697</td>
      <td>0.3948</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0518</td>
      <td>0.4440</td>
      <td>0.1922</td>
      <td>0.4448</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0542</td>
      <td>0.4931</td>
      <td>0.2117</td>
      <td>0.4939</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0595</td>
      <td>0.5425</td>
      <td>0.2314</td>
      <td>0.5432</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0653</td>
      <td>0.5922</td>
      <td>0.2566</td>
      <td>0.5927</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0678</td>
      <td>0.6409</td>
      <td>0.2750</td>
      <td>0.6414</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0695</td>
      <td>0.6908</td>
      <td>0.2944</td>
      <td>0.6912</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0697</td>
      <td>0.7403</td>
      <td>0.3139</td>
      <td>0.7407</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0681</td>
      <td>0.7890</td>
      <td>0.3317</td>
      <td>0.7893</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0683</td>
      <td>0.8388</td>
      <td>0.3558</td>
      <td>0.8390</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0568</td>
      <td>0.8882</td>
      <td>0.3755</td>
      <td>0.8883</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0418</td>
      <td>0.4087</td>
      <td>0.1752</td>
      <td>0.4092</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_10_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_10_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_10_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_10_acc.png)


## 20% positives
> train: [0.7999712 0.2000288]  
> validation: [0.7999712 0.2000288]  
> bin_sld: 312.036s  
> mul_sld: 63.355s  
> bin_sld_gs: 1039.328s  
> mul_sld_gs: 562.297s  
> bin_sld_gsq: 402.075s  
> bin_pacc: 297.187s  
> mul_pacc: 36.589s  
> binmc_pacc: 296.718s  
> mulmc_pacc: 36.777s  
> binne_pacc: 285.556s  
> mulne_pacc: 47.832s  
> bin_pacc_gs: 636.796s  
> mul_pacc_gs: 141.445s  
> bin_cc: 293.873s  
> mul_cc: 46.722s  
> kfcv: 30.476s  
> ref: 26.348s  
> atc_mc: 22.299s  
> atc_ne: 41.782s  
> doc_feat: 11.276s  
> tot: 1040.077s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0094</td>
      <td>0.0149</td>
      <td>0.0035</td>
      <td>0.0039</td>
      <td>0.1502</td>
      <td>0.0286</td>
      <td>0.1342</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0106</td>
      <td>0.0135</td>
      <td>0.0118</td>
      <td>0.0101</td>
      <td>0.1150</td>
      <td>0.0195</td>
      <td>0.0992</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0112</td>
      <td>0.0126</td>
      <td>0.0120</td>
      <td>0.0105</td>
      <td>0.0793</td>
      <td>0.0114</td>
      <td>0.0636</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0121</td>
      <td>0.0112</td>
      <td>0.0128</td>
      <td>0.0098</td>
      <td>0.0437</td>
      <td>0.0075</td>
      <td>0.0281</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0130</td>
      <td>0.0115</td>
      <td>0.0126</td>
      <td>0.0104</td>
      <td>0.0096</td>
      <td>0.0108</td>
      <td>0.0077</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0129</td>
      <td>0.0120</td>
      <td>0.0130</td>
      <td>0.0119</td>
      <td>0.0266</td>
      <td>0.0177</td>
      <td>0.0420</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0132</td>
      <td>0.0119</td>
      <td>0.0132</td>
      <td>0.0122</td>
      <td>0.0594</td>
      <td>0.0238</td>
      <td>0.0747</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0142</td>
      <td>0.0119</td>
      <td>0.0140</td>
      <td>0.0135</td>
      <td>0.0975</td>
      <td>0.0367</td>
      <td>0.1127</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0142</td>
      <td>0.0105</td>
      <td>0.0166</td>
      <td>0.0130</td>
      <td>0.1322</td>
      <td>0.0448</td>
      <td>0.1473</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0137</td>
      <td>0.0099</td>
      <td>0.0153</td>
      <td>0.0129</td>
      <td>0.1669</td>
      <td>0.0538</td>
      <td>0.1819</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0157</td>
      <td>0.0117</td>
      <td>0.0176</td>
      <td>0.0146</td>
      <td>0.2017</td>
      <td>0.0640</td>
      <td>0.2166</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0144</td>
      <td>0.0108</td>
      <td>0.0168</td>
      <td>0.0143</td>
      <td>0.2384</td>
      <td>0.0729</td>
      <td>0.2532</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0166</td>
      <td>0.0107</td>
      <td>0.0204</td>
      <td>0.0156</td>
      <td>0.2724</td>
      <td>0.0810</td>
      <td>0.2871</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0165</td>
      <td>0.0118</td>
      <td>0.0203</td>
      <td>0.0163</td>
      <td>0.3081</td>
      <td>0.0895</td>
      <td>0.3226</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0173</td>
      <td>0.0120</td>
      <td>0.0214</td>
      <td>0.0160</td>
      <td>0.3454</td>
      <td>0.1031</td>
      <td>0.3599</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0158</td>
      <td>0.0093</td>
      <td>0.0190</td>
      <td>0.0150</td>
      <td>0.3770</td>
      <td>0.1092</td>
      <td>0.3914</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0177</td>
      <td>0.0114</td>
      <td>0.0234</td>
      <td>0.0174</td>
      <td>0.4131</td>
      <td>0.1170</td>
      <td>0.4274</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0222</td>
      <td>0.0117</td>
      <td>0.0277</td>
      <td>0.0199</td>
      <td>0.4482</td>
      <td>0.1265</td>
      <td>0.4623</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0199</td>
      <td>0.0111</td>
      <td>0.0267</td>
      <td>0.0208</td>
      <td>0.4831</td>
      <td>0.1322</td>
      <td>0.4972</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0231</td>
      <td>0.0115</td>
      <td>0.0279</td>
      <td>0.0204</td>
      <td>0.5182</td>
      <td>0.1443</td>
      <td>0.5321</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0141</td>
      <td>0.0121</td>
      <td>0.0179</td>
      <td>0.0200</td>
      <td>0.5512</td>
      <td>0.1494</td>
      <td>0.5650</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0151</td>
      <td>0.0116</td>
      <td>0.0173</td>
      <td>0.0142</td>
      <td>0.2399</td>
      <td>0.0687</td>
      <td>0.2479</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_20_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_20_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_20_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_20_acc.png)


## 30% positives
> train: [0.7000288 0.2999712]  
> validation: [0.7000288 0.2999712]  
> bin_sld: 302.903s  
> mul_sld: 59.883s  
> bin_sld_gs: 1013.188s  
> mul_sld_gs: 556.087s  
> bin_sld_gsq: 413.115s  
> bin_pacc: 294.712s  
> mul_pacc: 45.954s  
> binmc_pacc: 300.950s  
> mulmc_pacc: 45.939s  
> binne_pacc: 289.350s  
> mulne_pacc: 38.137s  
> bin_pacc_gs: 643.072s  
> mul_pacc_gs: 149.582s  
> bin_cc: 298.684s  
> mul_cc: 43.908s  
> kfcv: 37.403s  
> ref: 28.756s  
> atc_mc: 39.420s  
> atc_ne: 24.347s  
> doc_feat: 22.815s  
> tot: 1014.314s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0060</td>
      <td>0.0135</td>
      <td>0.0182</td>
      <td>0.0103</td>
      <td>0.1183</td>
      <td>0.0279</td>
      <td>0.1152</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0101</td>
      <td>0.0130</td>
      <td>0.0219</td>
      <td>0.0124</td>
      <td>0.0990</td>
      <td>0.0220</td>
      <td>0.0959</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0120</td>
      <td>0.0123</td>
      <td>0.0195</td>
      <td>0.0128</td>
      <td>0.0781</td>
      <td>0.0157</td>
      <td>0.0752</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0126</td>
      <td>0.0118</td>
      <td>0.0204</td>
      <td>0.0136</td>
      <td>0.0569</td>
      <td>0.0111</td>
      <td>0.0541</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0136</td>
      <td>0.0128</td>
      <td>0.0186</td>
      <td>0.0153</td>
      <td>0.0387</td>
      <td>0.0108</td>
      <td>0.0360</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0163</td>
      <td>0.0144</td>
      <td>0.0215</td>
      <td>0.0185</td>
      <td>0.0185</td>
      <td>0.0106</td>
      <td>0.0159</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0146</td>
      <td>0.0120</td>
      <td>0.0168</td>
      <td>0.0154</td>
      <td>0.0073</td>
      <td>0.0105</td>
      <td>0.0079</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0185</td>
      <td>0.0132</td>
      <td>0.0201</td>
      <td>0.0194</td>
      <td>0.0220</td>
      <td>0.0143</td>
      <td>0.0245</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0197</td>
      <td>0.0137</td>
      <td>0.0185</td>
      <td>0.0211</td>
      <td>0.0435</td>
      <td>0.0199</td>
      <td>0.0459</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0189</td>
      <td>0.0125</td>
      <td>0.0185</td>
      <td>0.0223</td>
      <td>0.0627</td>
      <td>0.0235</td>
      <td>0.0650</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0226</td>
      <td>0.0141</td>
      <td>0.0193</td>
      <td>0.0237</td>
      <td>0.0830</td>
      <td>0.0301</td>
      <td>0.0852</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0225</td>
      <td>0.0117</td>
      <td>0.0163</td>
      <td>0.0212</td>
      <td>0.1028</td>
      <td>0.0355</td>
      <td>0.1050</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0247</td>
      <td>0.0129</td>
      <td>0.0177</td>
      <td>0.0255</td>
      <td>0.1227</td>
      <td>0.0404</td>
      <td>0.1247</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0271</td>
      <td>0.0136</td>
      <td>0.0174</td>
      <td>0.0266</td>
      <td>0.1421</td>
      <td>0.0442</td>
      <td>0.1441</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0315</td>
      <td>0.0145</td>
      <td>0.0176</td>
      <td>0.0283</td>
      <td>0.1632</td>
      <td>0.0517</td>
      <td>0.1651</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0325</td>
      <td>0.0147</td>
      <td>0.0166</td>
      <td>0.0290</td>
      <td>0.1833</td>
      <td>0.0584</td>
      <td>0.1851</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0353</td>
      <td>0.0164</td>
      <td>0.0185</td>
      <td>0.0322</td>
      <td>0.2015</td>
      <td>0.0608</td>
      <td>0.2033</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0409</td>
      <td>0.0168</td>
      <td>0.0179</td>
      <td>0.0320</td>
      <td>0.2215</td>
      <td>0.0669</td>
      <td>0.2232</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0436</td>
      <td>0.0188</td>
      <td>0.0201</td>
      <td>0.0355</td>
      <td>0.2391</td>
      <td>0.0698</td>
      <td>0.2406</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0455</td>
      <td>0.0203</td>
      <td>0.0172</td>
      <td>0.0332</td>
      <td>0.2625</td>
      <td>0.0804</td>
      <td>0.2639</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0415</td>
      <td>0.0296</td>
      <td>0.0054</td>
      <td>0.0221</td>
      <td>0.2779</td>
      <td>0.0817</td>
      <td>0.2793</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0243</td>
      <td>0.0149</td>
      <td>0.0180</td>
      <td>0.0224</td>
      <td>0.1212</td>
      <td>0.0374</td>
      <td>0.1217</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_30_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_30_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_30_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_30_acc.png)


## 40% positives
> train: [0.5999424 0.4000576]  
> validation: [0.60008641 0.39991359]  
> bin_sld: 298.351s  
> mul_sld: 51.270s  
> bin_sld_gs: 1033.280s  
> mul_sld_gs: 546.452s  
> bin_sld_gsq: 421.394s  
> bin_pacc: 298.508s  
> mul_pacc: 34.186s  
> binmc_pacc: 289.145s  
> mulmc_pacc: 42.455s  
> binne_pacc: 296.554s  
> mulne_pacc: 45.269s  
> bin_pacc_gs: 645.096s  
> mul_pacc_gs: 145.337s  
> bin_cc: 300.133s  
> mul_cc: 42.690s  
> kfcv: 36.406s  
> ref: 22.353s  
> atc_mc: 38.798s  
> atc_ne: 34.811s  
> doc_feat: 13.850s  
> tot: 1034.124s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0088</td>
      <td>0.0137</td>
      <td>0.0080</td>
      <td>0.0108</td>
      <td>0.0693</td>
      <td>0.0207</td>
      <td>0.0581</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0135</td>
      <td>0.0159</td>
      <td>0.0129</td>
      <td>0.0128</td>
      <td>0.0617</td>
      <td>0.0204</td>
      <td>0.0505</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0148</td>
      <td>0.0169</td>
      <td>0.0134</td>
      <td>0.0141</td>
      <td>0.0515</td>
      <td>0.0160</td>
      <td>0.0404</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0163</td>
      <td>0.0197</td>
      <td>0.0162</td>
      <td>0.0160</td>
      <td>0.0436</td>
      <td>0.0151</td>
      <td>0.0325</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0180</td>
      <td>0.0204</td>
      <td>0.0182</td>
      <td>0.0165</td>
      <td>0.0357</td>
      <td>0.0133</td>
      <td>0.0247</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0159</td>
      <td>0.0203</td>
      <td>0.0159</td>
      <td>0.0155</td>
      <td>0.0265</td>
      <td>0.0099</td>
      <td>0.0159</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0176</td>
      <td>0.0199</td>
      <td>0.0175</td>
      <td>0.0142</td>
      <td>0.0188</td>
      <td>0.0107</td>
      <td>0.0102</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0218</td>
      <td>0.0242</td>
      <td>0.0212</td>
      <td>0.0185</td>
      <td>0.0115</td>
      <td>0.0089</td>
      <td>0.0072</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0233</td>
      <td>0.0250</td>
      <td>0.0228</td>
      <td>0.0187</td>
      <td>0.0081</td>
      <td>0.0099</td>
      <td>0.0126</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0234</td>
      <td>0.0242</td>
      <td>0.0223</td>
      <td>0.0172</td>
      <td>0.0107</td>
      <td>0.0109</td>
      <td>0.0183</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0270</td>
      <td>0.0273</td>
      <td>0.0277</td>
      <td>0.0214</td>
      <td>0.0158</td>
      <td>0.0115</td>
      <td>0.0256</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0249</td>
      <td>0.0238</td>
      <td>0.0250</td>
      <td>0.0202</td>
      <td>0.0267</td>
      <td>0.0117</td>
      <td>0.0374</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0299</td>
      <td>0.0271</td>
      <td>0.0294</td>
      <td>0.0207</td>
      <td>0.0329</td>
      <td>0.0135</td>
      <td>0.0435</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0291</td>
      <td>0.0266</td>
      <td>0.0293</td>
      <td>0.0223</td>
      <td>0.0403</td>
      <td>0.0142</td>
      <td>0.0509</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0313</td>
      <td>0.0268</td>
      <td>0.0308</td>
      <td>0.0225</td>
      <td>0.0523</td>
      <td>0.0168</td>
      <td>0.0627</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0335</td>
      <td>0.0282</td>
      <td>0.0332</td>
      <td>0.0246</td>
      <td>0.0579</td>
      <td>0.0182</td>
      <td>0.0683</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0326</td>
      <td>0.0277</td>
      <td>0.0360</td>
      <td>0.0285</td>
      <td>0.0669</td>
      <td>0.0198</td>
      <td>0.0772</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0376</td>
      <td>0.0304</td>
      <td>0.0385</td>
      <td>0.0296</td>
      <td>0.0753</td>
      <td>0.0229</td>
      <td>0.0857</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0350</td>
      <td>0.0281</td>
      <td>0.0377</td>
      <td>0.0310</td>
      <td>0.0827</td>
      <td>0.0272</td>
      <td>0.0930</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0388</td>
      <td>0.0298</td>
      <td>0.0405</td>
      <td>0.0314</td>
      <td>0.0931</td>
      <td>0.0283</td>
      <td>0.1034</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0358</td>
      <td>0.0376</td>
      <td>0.0370</td>
      <td>0.0352</td>
      <td>0.0997</td>
      <td>0.0310</td>
      <td>0.1099</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0252</td>
      <td>0.0244</td>
      <td>0.0254</td>
      <td>0.0210</td>
      <td>0.0467</td>
      <td>0.0167</td>
      <td>0.0490</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_40_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_40_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_40_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_40_acc.png)


## 50% positives
> train: [0.5 0.5]  
> validation: [0.5 0.5]  
> bin_sld: 303.489s  
> mul_sld: 50.183s  
> bin_sld_gs: 1015.361s  
> mul_sld_gs: 552.539s  
> bin_sld_gsq: 418.096s  
> bin_pacc: 296.283s  
> mul_pacc: 32.547s  
> binmc_pacc: 299.487s  
> mulmc_pacc: 40.506s  
> binne_pacc: 293.984s  
> mulne_pacc: 39.974s  
> bin_pacc_gs: 643.158s  
> mul_pacc_gs: 146.732s  
> bin_cc: 292.672s  
> mul_cc: 37.348s  
> kfcv: 29.093s  
> ref: 16.306s  
> atc_mc: 25.874s  
> atc_ne: 28.890s  
> doc_feat: 11.601s  
> tot: 1016.064s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0085</td>
      <td>0.0080</td>
      <td>0.0061</td>
      <td>0.0111</td>
      <td>0.0084</td>
      <td>0.0100</td>
      <td>0.0094</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0114</td>
      <td>0.0086</td>
      <td>0.0190</td>
      <td>0.0144</td>
      <td>0.0084</td>
      <td>0.0090</td>
      <td>0.0096</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0145</td>
      <td>0.0093</td>
      <td>0.0205</td>
      <td>0.0141</td>
      <td>0.0089</td>
      <td>0.0096</td>
      <td>0.0097</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0154</td>
      <td>0.0106</td>
      <td>0.0212</td>
      <td>0.0141</td>
      <td>0.0082</td>
      <td>0.0085</td>
      <td>0.0089</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0151</td>
      <td>0.0108</td>
      <td>0.0213</td>
      <td>0.0169</td>
      <td>0.0087</td>
      <td>0.0086</td>
      <td>0.0092</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0156</td>
      <td>0.0108</td>
      <td>0.0209</td>
      <td>0.0164</td>
      <td>0.0083</td>
      <td>0.0094</td>
      <td>0.0089</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0161</td>
      <td>0.0128</td>
      <td>0.0242</td>
      <td>0.0165</td>
      <td>0.0089</td>
      <td>0.0098</td>
      <td>0.0092</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0188</td>
      <td>0.0131</td>
      <td>0.0262</td>
      <td>0.0152</td>
      <td>0.0077</td>
      <td>0.0094</td>
      <td>0.0080</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0188</td>
      <td>0.0128</td>
      <td>0.0266</td>
      <td>0.0157</td>
      <td>0.0087</td>
      <td>0.0101</td>
      <td>0.0093</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0183</td>
      <td>0.0132</td>
      <td>0.0265</td>
      <td>0.0169</td>
      <td>0.0095</td>
      <td>0.0103</td>
      <td>0.0097</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0231</td>
      <td>0.0173</td>
      <td>0.0316</td>
      <td>0.0208</td>
      <td>0.0097</td>
      <td>0.0124</td>
      <td>0.0097</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0198</td>
      <td>0.0172</td>
      <td>0.0276</td>
      <td>0.0189</td>
      <td>0.0075</td>
      <td>0.0113</td>
      <td>0.0074</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0209</td>
      <td>0.0176</td>
      <td>0.0299</td>
      <td>0.0214</td>
      <td>0.0080</td>
      <td>0.0110</td>
      <td>0.0080</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0196</td>
      <td>0.0167</td>
      <td>0.0267</td>
      <td>0.0200</td>
      <td>0.0076</td>
      <td>0.0112</td>
      <td>0.0073</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0222</td>
      <td>0.0189</td>
      <td>0.0310</td>
      <td>0.0202</td>
      <td>0.0082</td>
      <td>0.0109</td>
      <td>0.0077</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0227</td>
      <td>0.0203</td>
      <td>0.0313</td>
      <td>0.0203</td>
      <td>0.0095</td>
      <td>0.0109</td>
      <td>0.0086</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0215</td>
      <td>0.0192</td>
      <td>0.0324</td>
      <td>0.0215</td>
      <td>0.0102</td>
      <td>0.0100</td>
      <td>0.0092</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0246</td>
      <td>0.0210</td>
      <td>0.0328</td>
      <td>0.0207</td>
      <td>0.0100</td>
      <td>0.0107</td>
      <td>0.0092</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0211</td>
      <td>0.0204</td>
      <td>0.0324</td>
      <td>0.0215</td>
      <td>0.0111</td>
      <td>0.0114</td>
      <td>0.0100</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0252</td>
      <td>0.0243</td>
      <td>0.0360</td>
      <td>0.0226</td>
      <td>0.0098</td>
      <td>0.0119</td>
      <td>0.0087</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0266</td>
      <td>0.0292</td>
      <td>0.0316</td>
      <td>0.0230</td>
      <td>0.0113</td>
      <td>0.0107</td>
      <td>0.0100</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0190</td>
      <td>0.0158</td>
      <td>0.0265</td>
      <td>0.0182</td>
      <td>0.0090</td>
      <td>0.0103</td>
      <td>0.0089</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_50_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_50_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_50_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_50_acc.png)


## 60% positives
> train: [0.39991359 0.60008641]  
> validation: [0.4000576 0.5999424]  
> bin_sld: 296.102s  
> mul_sld: 56.083s  
> bin_sld_gs: 1060.976s  
> mul_sld_gs: 551.619s  
> bin_sld_gsq: 425.664s  
> bin_pacc: 293.321s  
> mul_pacc: 44.690s  
> binmc_pacc: 296.400s  
> mulmc_pacc: 38.640s  
> binne_pacc: 288.195s  
> mulne_pacc: 44.768s  
> bin_pacc_gs: 645.660s  
> mul_pacc_gs: 146.232s  
> bin_cc: 299.198s  
> mul_cc: 41.642s  
> kfcv: 21.885s  
> ref: 26.261s  
> atc_mc: 37.391s  
> atc_ne: 35.750s  
> doc_feat: 19.963s  
> tot: 1061.773s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0028</td>
      <td>0.0064</td>
      <td>0.0078</td>
      <td>0.0130</td>
      <td>0.1073</td>
      <td>0.0238</td>
      <td>0.1133</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0139</td>
      <td>0.0081</td>
      <td>0.0152</td>
      <td>0.0141</td>
      <td>0.0966</td>
      <td>0.0205</td>
      <td>0.1026</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0161</td>
      <td>0.0098</td>
      <td>0.0162</td>
      <td>0.0144</td>
      <td>0.0909</td>
      <td>0.0209</td>
      <td>0.0970</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0188</td>
      <td>0.0111</td>
      <td>0.0147</td>
      <td>0.0159</td>
      <td>0.0765</td>
      <td>0.0139</td>
      <td>0.0826</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0194</td>
      <td>0.0131</td>
      <td>0.0155</td>
      <td>0.0186</td>
      <td>0.0693</td>
      <td>0.0166</td>
      <td>0.0755</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0199</td>
      <td>0.0137</td>
      <td>0.0170</td>
      <td>0.0163</td>
      <td>0.0582</td>
      <td>0.0133</td>
      <td>0.0644</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0189</td>
      <td>0.0152</td>
      <td>0.0164</td>
      <td>0.0186</td>
      <td>0.0479</td>
      <td>0.0127</td>
      <td>0.0542</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0242</td>
      <td>0.0195</td>
      <td>0.0194</td>
      <td>0.0208</td>
      <td>0.0388</td>
      <td>0.0116</td>
      <td>0.0451</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0257</td>
      <td>0.0214</td>
      <td>0.0196</td>
      <td>0.0217</td>
      <td>0.0307</td>
      <td>0.0120</td>
      <td>0.0370</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0235</td>
      <td>0.0206</td>
      <td>0.0177</td>
      <td>0.0207</td>
      <td>0.0198</td>
      <td>0.0118</td>
      <td>0.0259</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0288</td>
      <td>0.0257</td>
      <td>0.0224</td>
      <td>0.0253</td>
      <td>0.0128</td>
      <td>0.0122</td>
      <td>0.0178</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0266</td>
      <td>0.0247</td>
      <td>0.0203</td>
      <td>0.0244</td>
      <td>0.0086</td>
      <td>0.0127</td>
      <td>0.0099</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0296</td>
      <td>0.0272</td>
      <td>0.0228</td>
      <td>0.0269</td>
      <td>0.0102</td>
      <td>0.0134</td>
      <td>0.0081</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0272</td>
      <td>0.0265</td>
      <td>0.0216</td>
      <td>0.0260</td>
      <td>0.0197</td>
      <td>0.0147</td>
      <td>0.0137</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0325</td>
      <td>0.0329</td>
      <td>0.0241</td>
      <td>0.0338</td>
      <td>0.0283</td>
      <td>0.0193</td>
      <td>0.0219</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0316</td>
      <td>0.0328</td>
      <td>0.0241</td>
      <td>0.0334</td>
      <td>0.0385</td>
      <td>0.0206</td>
      <td>0.0320</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0285</td>
      <td>0.0315</td>
      <td>0.0234</td>
      <td>0.0318</td>
      <td>0.0503</td>
      <td>0.0241</td>
      <td>0.0437</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0292</td>
      <td>0.0333</td>
      <td>0.0210</td>
      <td>0.0344</td>
      <td>0.0566</td>
      <td>0.0237</td>
      <td>0.0500</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0281</td>
      <td>0.0332</td>
      <td>0.0218</td>
      <td>0.0349</td>
      <td>0.0687</td>
      <td>0.0286</td>
      <td>0.0620</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0320</td>
      <td>0.0382</td>
      <td>0.0250</td>
      <td>0.0417</td>
      <td>0.0780</td>
      <td>0.0316</td>
      <td>0.0713</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0286</td>
      <td>0.0395</td>
      <td>0.0234</td>
      <td>0.0412</td>
      <td>0.0876</td>
      <td>0.0343</td>
      <td>0.0809</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0241</td>
      <td>0.0231</td>
      <td>0.0195</td>
      <td>0.0251</td>
      <td>0.0522</td>
      <td>0.0187</td>
      <td>0.0528</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_60_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_60_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_60_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_60_acc.png)


## 70% positives
> train: [0.2999712 0.7000288]  
> validation: [0.2999712 0.7000288]  
> bin_sld: 303.930s  
> mul_sld: 40.110s  
> bin_sld_gs: 1015.962s  
> mul_sld_gs: 557.286s  
> bin_sld_gsq: 414.452s  
> bin_pacc: 292.176s  
> mul_pacc: 47.800s  
> binmc_pacc: 302.570s  
> mulmc_pacc: 44.408s  
> binne_pacc: 293.019s  
> mulne_pacc: 43.783s  
> bin_pacc_gs: 648.943s  
> mul_pacc_gs: 150.210s  
> bin_cc: 288.613s  
> mul_cc: 38.672s  
> kfcv: 32.153s  
> ref: 34.374s  
> atc_mc: 38.938s  
> atc_ne: 40.313s  
> doc_feat: 20.243s  
> tot: 1016.774s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0371</td>
      <td>0.0064</td>
      <td>0.0180</td>
      <td>0.0284</td>
      <td>0.2598</td>
      <td>0.0699</td>
      <td>0.2720</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0384</td>
      <td>0.0087</td>
      <td>0.0170</td>
      <td>0.0258</td>
      <td>0.2399</td>
      <td>0.0635</td>
      <td>0.2522</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0392</td>
      <td>0.0089</td>
      <td>0.0188</td>
      <td>0.0248</td>
      <td>0.2224</td>
      <td>0.0617</td>
      <td>0.2348</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0361</td>
      <td>0.0109</td>
      <td>0.0164</td>
      <td>0.0213</td>
      <td>0.2010</td>
      <td>0.0528</td>
      <td>0.2134</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0337</td>
      <td>0.0112</td>
      <td>0.0171</td>
      <td>0.0204</td>
      <td>0.1842</td>
      <td>0.0507</td>
      <td>0.1967</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0313</td>
      <td>0.0105</td>
      <td>0.0162</td>
      <td>0.0208</td>
      <td>0.1607</td>
      <td>0.0430</td>
      <td>0.1733</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0280</td>
      <td>0.0100</td>
      <td>0.0156</td>
      <td>0.0191</td>
      <td>0.1416</td>
      <td>0.0386</td>
      <td>0.1543</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0219</td>
      <td>0.0104</td>
      <td>0.0131</td>
      <td>0.0160</td>
      <td>0.1228</td>
      <td>0.0338</td>
      <td>0.1356</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0190</td>
      <td>0.0106</td>
      <td>0.0144</td>
      <td>0.0165</td>
      <td>0.1045</td>
      <td>0.0306</td>
      <td>0.1172</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0177</td>
      <td>0.0115</td>
      <td>0.0154</td>
      <td>0.0172</td>
      <td>0.0834</td>
      <td>0.0265</td>
      <td>0.0962</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0144</td>
      <td>0.0128</td>
      <td>0.0153</td>
      <td>0.0170</td>
      <td>0.0646</td>
      <td>0.0207</td>
      <td>0.0776</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0101</td>
      <td>0.0117</td>
      <td>0.0121</td>
      <td>0.0132</td>
      <td>0.0450</td>
      <td>0.0167</td>
      <td>0.0580</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0096</td>
      <td>0.0111</td>
      <td>0.0145</td>
      <td>0.0148</td>
      <td>0.0278</td>
      <td>0.0144</td>
      <td>0.0407</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0107</td>
      <td>0.0135</td>
      <td>0.0140</td>
      <td>0.0160</td>
      <td>0.0088</td>
      <td>0.0118</td>
      <td>0.0178</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0106</td>
      <td>0.0136</td>
      <td>0.0109</td>
      <td>0.0145</td>
      <td>0.0145</td>
      <td>0.0100</td>
      <td>0.0072</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0113</td>
      <td>0.0130</td>
      <td>0.0134</td>
      <td>0.0137</td>
      <td>0.0333</td>
      <td>0.0090</td>
      <td>0.0201</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0135</td>
      <td>0.0128</td>
      <td>0.0148</td>
      <td>0.0154</td>
      <td>0.0541</td>
      <td>0.0125</td>
      <td>0.0407</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0171</td>
      <td>0.0151</td>
      <td>0.0171</td>
      <td>0.0184</td>
      <td>0.0721</td>
      <td>0.0141</td>
      <td>0.0587</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0190</td>
      <td>0.0145</td>
      <td>0.0168</td>
      <td>0.0173</td>
      <td>0.0931</td>
      <td>0.0185</td>
      <td>0.0796</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0283</td>
      <td>0.0181</td>
      <td>0.0266</td>
      <td>0.0226</td>
      <td>0.1124</td>
      <td>0.0250</td>
      <td>0.0988</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0404</td>
      <td>0.0202</td>
      <td>0.0205</td>
      <td>0.0238</td>
      <td>0.1322</td>
      <td>0.0288</td>
      <td>0.1186</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0232</td>
      <td>0.0122</td>
      <td>0.0161</td>
      <td>0.0189</td>
      <td>0.1133</td>
      <td>0.0311</td>
      <td>0.1173</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_70_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_70_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_70_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_70_acc.png)


## 80% positives
> train: [0.2000288 0.7999712]  
> validation: [0.2000288 0.7999712]  
> bin_sld: 302.697s  
> mul_sld: 51.427s  
> bin_sld_gs: 1033.432s  
> mul_sld_gs: 553.840s  
> bin_sld_gsq: 405.825s  
> bin_pacc: 296.888s  
> mul_pacc: 34.390s  
> binmc_pacc: 300.681s  
> mulmc_pacc: 44.972s  
> binne_pacc: 284.115s  
> mulne_pacc: 37.204s  
> bin_pacc_gs: 637.277s  
> mul_pacc_gs: 146.708s  
> bin_cc: 297.171s  
> mul_cc: 42.941s  
> kfcv: 27.967s  
> ref: 32.469s  
> atc_mc: 22.965s  
> atc_ne: 39.769s  
> doc_feat: 19.964s  
> tot: 1034.198s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0147</td>
      <td>0.0059</td>
      <td>0.0387</td>
      <td>0.0267</td>
      <td>0.4991</td>
      <td>0.1270</td>
      <td>0.5060</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0332</td>
      <td>0.0087</td>
      <td>0.0323</td>
      <td>0.0201</td>
      <td>0.4710</td>
      <td>0.1198</td>
      <td>0.4780</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0221</td>
      <td>0.0082</td>
      <td>0.0321</td>
      <td>0.0209</td>
      <td>0.4396</td>
      <td>0.1165</td>
      <td>0.4468</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0242</td>
      <td>0.0091</td>
      <td>0.0267</td>
      <td>0.0158</td>
      <td>0.4063</td>
      <td>0.1057</td>
      <td>0.4135</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0245</td>
      <td>0.0106</td>
      <td>0.0256</td>
      <td>0.0174</td>
      <td>0.3744</td>
      <td>0.0976</td>
      <td>0.3817</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0240</td>
      <td>0.0119</td>
      <td>0.0224</td>
      <td>0.0158</td>
      <td>0.3419</td>
      <td>0.0873</td>
      <td>0.3493</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0232</td>
      <td>0.0101</td>
      <td>0.0201</td>
      <td>0.0134</td>
      <td>0.3098</td>
      <td>0.0804</td>
      <td>0.3173</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0250</td>
      <td>0.0103</td>
      <td>0.0161</td>
      <td>0.0127</td>
      <td>0.2767</td>
      <td>0.0719</td>
      <td>0.2843</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0238</td>
      <td>0.0121</td>
      <td>0.0161</td>
      <td>0.0129</td>
      <td>0.2452</td>
      <td>0.0668</td>
      <td>0.2529</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0221</td>
      <td>0.0121</td>
      <td>0.0148</td>
      <td>0.0131</td>
      <td>0.2111</td>
      <td>0.0571</td>
      <td>0.2189</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0240</td>
      <td>0.0128</td>
      <td>0.0121</td>
      <td>0.0139</td>
      <td>0.1813</td>
      <td>0.0507</td>
      <td>0.1892</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0239</td>
      <td>0.0118</td>
      <td>0.0106</td>
      <td>0.0126</td>
      <td>0.1485</td>
      <td>0.0430</td>
      <td>0.1565</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0295</td>
      <td>0.0132</td>
      <td>0.0099</td>
      <td>0.0161</td>
      <td>0.1188</td>
      <td>0.0363</td>
      <td>0.1269</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0214</td>
      <td>0.0134</td>
      <td>0.0104</td>
      <td>0.0147</td>
      <td>0.0831</td>
      <td>0.0247</td>
      <td>0.0912</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0260</td>
      <td>0.0149</td>
      <td>0.0114</td>
      <td>0.0186</td>
      <td>0.0519</td>
      <td>0.0184</td>
      <td>0.0602</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0307</td>
      <td>0.0149</td>
      <td>0.0122</td>
      <td>0.0204</td>
      <td>0.0202</td>
      <td>0.0119</td>
      <td>0.0286</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0308</td>
      <td>0.0152</td>
      <td>0.0137</td>
      <td>0.0209</td>
      <td>0.0129</td>
      <td>0.0099</td>
      <td>0.0068</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0330</td>
      <td>0.0144</td>
      <td>0.0161</td>
      <td>0.0233</td>
      <td>0.0438</td>
      <td>0.0101</td>
      <td>0.0352</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0311</td>
      <td>0.0128</td>
      <td>0.0168</td>
      <td>0.0225</td>
      <td>0.0767</td>
      <td>0.0152</td>
      <td>0.0680</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0336</td>
      <td>0.0146</td>
      <td>0.0220</td>
      <td>0.0276</td>
      <td>0.1089</td>
      <td>0.0224</td>
      <td>0.1001</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0293</td>
      <td>0.0125</td>
      <td>0.0230</td>
      <td>0.0281</td>
      <td>0.1408</td>
      <td>0.0294</td>
      <td>0.1320</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0262</td>
      <td>0.0119</td>
      <td>0.0192</td>
      <td>0.0185</td>
      <td>0.2172</td>
      <td>0.0572</td>
      <td>0.2211</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_80_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_80_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_80_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_80_acc.png)


## 90% positives
> train: [0.0999424 0.9000576]  
> validation: [0.10008641 0.89991359]  
> bin_sld: 170.505s  
> mul_sld: 31.294s  
> mul_sld_gs: 380.780s  
> bin_cc: 164.518s  
> mul_cc: 20.620s  
> kfcv: 15.634s  
> ref: 13.757s  
> atc_mc: 18.010s  
> atc_ne: 18.026s  
> doc_feat: 10.161s  
> tot: 381.531s  

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mul_sld_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0502</td>
      <td>0.8376</td>
      <td>0.2957</td>
      <td>0.8389</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0550</td>
      <td>0.7925</td>
      <td>0.2812</td>
      <td>0.7938</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0580</td>
      <td>0.7460</td>
      <td>0.2658</td>
      <td>0.7475</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0561</td>
      <td>0.6986</td>
      <td>0.2470</td>
      <td>0.7001</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0549</td>
      <td>0.6517</td>
      <td>0.2327</td>
      <td>0.6533</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0514</td>
      <td>0.6056</td>
      <td>0.2133</td>
      <td>0.6072</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0490</td>
      <td>0.5577</td>
      <td>0.1958</td>
      <td>0.5595</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0448</td>
      <td>0.5117</td>
      <td>0.1800</td>
      <td>0.5135</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0421</td>
      <td>0.4644</td>
      <td>0.1661</td>
      <td>0.4663</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0370</td>
      <td>0.4167</td>
      <td>0.1473</td>
      <td>0.4187</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0324</td>
      <td>0.3701</td>
      <td>0.1307</td>
      <td>0.3722</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0276</td>
      <td>0.3232</td>
      <td>0.1150</td>
      <td>0.3254</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0237</td>
      <td>0.2776</td>
      <td>0.0995</td>
      <td>0.2798</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0171</td>
      <td>0.2296</td>
      <td>0.0798</td>
      <td>0.2319</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0108</td>
      <td>0.1832</td>
      <td>0.0650</td>
      <td>0.1856</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0084</td>
      <td>0.1372</td>
      <td>0.0479</td>
      <td>0.1396</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0084</td>
      <td>0.0900</td>
      <td>0.0312</td>
      <td>0.0926</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0087</td>
      <td>0.0438</td>
      <td>0.0160</td>
      <td>0.0464</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0111</td>
      <td>0.0038</td>
      <td>0.0060</td>
      <td>0.0022</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0168</td>
      <td>0.0501</td>
      <td>0.0171</td>
      <td>0.0474</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0167</td>
      <td>0.0970</td>
      <td>0.0339</td>
      <td>0.0941</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0324</td>
      <td>0.3851</td>
      <td>0.1365</td>
      <td>0.3865</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_90_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_90_acc.png)
![plot_diagonal](plot/diagonal_imdb_gs_vs_all_90_acc.png)
![plot_shift](plot/shift_imdb_gs_vs_all_90_acc.png)


## avg
### avg on train
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.0125</td>
      <td>0.0157</td>
      <td>0.0143</td>
      <td>0.0149</td>
      <td>0.2388</td>
      <td>0.0712</td>
      <td>0.2383</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>0.0188</td>
      <td>0.0163</td>
      <td>0.0186</td>
      <td>0.0157</td>
      <td>0.2149</td>
      <td>0.0635</td>
      <td>0.2145</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>0.0186</td>
      <td>0.0164</td>
      <td>0.0190</td>
      <td>0.0160</td>
      <td>0.1909</td>
      <td>0.0582</td>
      <td>0.1907</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>0.0194</td>
      <td>0.0164</td>
      <td>0.0184</td>
      <td>0.0152</td>
      <td>0.1760</td>
      <td>0.0541</td>
      <td>0.1759</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0196</td>
      <td>0.0171</td>
      <td>0.0184</td>
      <td>0.0165</td>
      <td>0.1634</td>
      <td>0.0541</td>
      <td>0.1649</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>0.0194</td>
      <td>0.0177</td>
      <td>0.0181</td>
      <td>0.0165</td>
      <td>0.1549</td>
      <td>0.0523</td>
      <td>0.1584</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0188</td>
      <td>0.0179</td>
      <td>0.0177</td>
      <td>0.0156</td>
      <td>0.1498</td>
      <td>0.0520</td>
      <td>0.1539</td>
    </tr>
    <tr>
      <th>0.35</th>
      <td>0.0206</td>
      <td>0.0194</td>
      <td>0.0186</td>
      <td>0.0166</td>
      <td>0.1483</td>
      <td>0.0529</td>
      <td>0.1531</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0206</td>
      <td>0.0203</td>
      <td>0.0192</td>
      <td>0.0171</td>
      <td>0.1482</td>
      <td>0.0545</td>
      <td>0.1540</td>
    </tr>
    <tr>
      <th>0.45</th>
      <td>0.0197</td>
      <td>0.0201</td>
      <td>0.0186</td>
      <td>0.0172</td>
      <td>0.1473</td>
      <td>0.0543</td>
      <td>0.1534</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0222</td>
      <td>0.0222</td>
      <td>0.0208</td>
      <td>0.0195</td>
      <td>0.1481</td>
      <td>0.0558</td>
      <td>0.1543</td>
    </tr>
    <tr>
      <th>0.55</th>
      <td>0.0203</td>
      <td>0.0212</td>
      <td>0.0184</td>
      <td>0.0178</td>
      <td>0.1494</td>
      <td>0.0568</td>
      <td>0.1553</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0230</td>
      <td>0.0220</td>
      <td>0.0207</td>
      <td>0.0201</td>
      <td>0.1515</td>
      <td>0.0579</td>
      <td>0.1570</td>
    </tr>
    <tr>
      <th>0.65</th>
      <td>0.0217</td>
      <td>0.0221</td>
      <td>0.0200</td>
      <td>0.0203</td>
      <td>0.1535</td>
      <td>0.0579</td>
      <td>0.1581</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0245</td>
      <td>0.0233</td>
      <td>0.0210</td>
      <td>0.0220</td>
      <td>0.1599</td>
      <td>0.0613</td>
      <td>0.1625</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.0254</td>
      <td>0.0233</td>
      <td>0.0214</td>
      <td>0.0224</td>
      <td>0.1664</td>
      <td>0.0623</td>
      <td>0.1684</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0257</td>
      <td>0.0236</td>
      <td>0.0232</td>
      <td>0.0240</td>
      <td>0.1766</td>
      <td>0.0644</td>
      <td>0.1769</td>
    </tr>
    <tr>
      <th>0.85</th>
      <td>0.0292</td>
      <td>0.0246</td>
      <td>0.0245</td>
      <td>0.0255</td>
      <td>0.1902</td>
      <td>0.0672</td>
      <td>0.1901</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>0.0283</td>
      <td>0.0242</td>
      <td>0.0246</td>
      <td>0.0262</td>
      <td>0.2052</td>
      <td>0.0712</td>
      <td>0.2047</td>
    </tr>
    <tr>
      <th>0.95</th>
      <td>0.0324</td>
      <td>0.0269</td>
      <td>0.0279</td>
      <td>0.0285</td>
      <td>0.2302</td>
      <td>0.0796</td>
      <td>0.2294</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.0309</td>
      <td>0.0282</td>
      <td>0.0227</td>
      <td>0.0276</td>
      <td>0.2540</td>
      <td>0.0861</td>
      <td>0.2531</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0225</td>
      <td>0.0209</td>
      <td>0.0203</td>
      <td>0.0198</td>
      <td>0.1770</td>
      <td>0.0613</td>
      <td>0.1794</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_avg_train_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_avg_train_acc.png)
### avg on test
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bin_sld_gs</th>
      <th>mul_sld_gs</th>
      <th>bin_pacc_gs</th>
      <th>mul_pacc_gs</th>
      <th>kfcv</th>
      <th>atc_mc</th>
      <th>doc_feat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.1</th>
      <td>NaN</td>
      <td>0.0418</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.4087</td>
      <td>0.1752</td>
      <td>0.4092</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>0.0151</td>
      <td>0.0116</td>
      <td>0.0173</td>
      <td>0.0142</td>
      <td>0.2399</td>
      <td>0.0687</td>
      <td>0.2479</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>0.0243</td>
      <td>0.0149</td>
      <td>0.0180</td>
      <td>0.0224</td>
      <td>0.1212</td>
      <td>0.0374</td>
      <td>0.1217</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>0.0252</td>
      <td>0.0244</td>
      <td>0.0254</td>
      <td>0.0210</td>
      <td>0.0467</td>
      <td>0.0167</td>
      <td>0.0490</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>0.0190</td>
      <td>0.0158</td>
      <td>0.0265</td>
      <td>0.0182</td>
      <td>0.0090</td>
      <td>0.0103</td>
      <td>0.0089</td>
    </tr>
    <tr>
      <th>0.6</th>
      <td>0.0241</td>
      <td>0.0231</td>
      <td>0.0195</td>
      <td>0.0251</td>
      <td>0.0522</td>
      <td>0.0187</td>
      <td>0.0528</td>
    </tr>
    <tr>
      <th>0.7</th>
      <td>0.0232</td>
      <td>0.0122</td>
      <td>0.0161</td>
      <td>0.0189</td>
      <td>0.1133</td>
      <td>0.0311</td>
      <td>0.1173</td>
    </tr>
    <tr>
      <th>0.8</th>
      <td>0.0262</td>
      <td>0.0119</td>
      <td>0.0192</td>
      <td>0.0185</td>
      <td>0.2172</td>
      <td>0.0572</td>
      <td>0.2211</td>
    </tr>
    <tr>
      <th>0.9</th>
      <td>NaN</td>
      <td>0.0324</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.3851</td>
      <td>0.1365</td>
      <td>0.3865</td>
    </tr>
    <tr>
      <th>avg</th>
      <td>0.0225</td>
      <td>0.0209</td>
      <td>0.0203</td>
      <td>0.0198</td>
      <td>0.1770</td>
      <td>0.0613</td>
      <td>0.1794</td>
    </tr>
  </tbody>
</table>

![plot_delta](plot/delta_imdb_gs_vs_all_avg_test_acc.png)
![plot_delta_stdev](plot/delta_stdev_imdb_gs_vs_all_avg_test_acc.png)
### avg dataset shift
![plot_shift](plot/shift_imdb_gs_vs_all_avg_acc.png)
