- [x] aggiungere media tabelle
- [x] plot; 3 tipi (appunti + email + garg)
- [x] sistemare kfcv baseline
- [x] aggiungere metodo con CC oltre SLD
- [x] prendere classe più popolosa di rcv1, togliere negativi fino a raggiungere 50/50; poi fare subsampling con 9 training prvalences (da 0.1-0.9 a 0.9-0.1)
- [x] variare parametro recalibration in SLD


- [x] fix grafico diagonal
    - seaborn example gallery
- [x] varianti recalib: bcts, SLD (provare exact_train_prev=False)
- [x] vedere cosa usa garg di validation size
- [x] per model selection testare il parametro c del classificatore, si esplora in np.logscale(-3,3, 7) oppure np.logscale(-4, 4, 9), parametro class_weight si esplora in None oppure "balanced"; va usato qp.model_selection.GridSearchQ in funzione di mae come errore, UPP come protocollo
    - qp.train_test_split per avere v_train e v_val
    - GridSearchQ(
        model: BaseQuantifier,
        param_grid: {
            'classifier__C': np.logspace(-3,3,7),
            'classifier__class_weight': [None, 'balanced'],
            'recalib': [None, 'bcts']
        },
        protocol: UPP(V_val, repeats=1000),
        error = qp.error.mae,
        refit=True,
        timeout=-1,
        n_jobs=-2,
        verbose=True).fit(V_tr)
- [x] plot collettivo, con sulla x lo shift e prenda in considerazione tutti i training set, facendo la media sui 9 casi (ogni line è un metodo), risultati non ottimizzati e ottimizzati
- [x] salvare il best score ottenuto da ogni applicazione di GridSearchQ
    - nel caso di bin fare media dei due best score
- [x] import baselines
