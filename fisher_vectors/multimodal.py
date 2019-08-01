def train(traj_files,
          all_labels,
          all_info,
          bag_idx,
          feature_idx,
          exp_path,
          svm_c=None,
          num_bags=100000,
          num_folds=4,
          balance=True,
          random_state=22,
          frame_scores_path=None,
          num_frames=36000,
          skip_train=False):

    # Cross-testing loop
    results = list()
    scores = [np.empty((len(l))) for l in all_labels]

    if frame_scores_path:
        frame_scores = np.empty((num_frames, len(all_info)))

    kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    for f, (train_idxs, test_idxs) in enumerate(kf.split(traj_files)):
        print('{:*^10}'.format(' FOLD {:d} '.format(f)))
        fold_results = {'fold': f, 'first_test_elem': test_idxs[0]}

        train_files = [traj_files[i] for i in train_idxs]
        test_files = [traj_files[i] for i in test_idxs]

        fold_path = os.path.join(exp_path, 'fold{:02d}'.format(f))
        if not os.path.exists(fold_path):
            os.mkdir(fold_path)

        train_fvs_path = os.path.join(fold_path, 'train_fvs.pkl')
        test_fvs_path = os.path.join(fold_path, 'test_fvs.pkl')

        if frame_scores_path:
            for i in test_idxs:
                assert int(len(all_labels[i])) == len(all_info[i])
            test_map = np.concatenate(
                [np.arange(0, len(all_info[i])) for i in test_idxs])

        X_train = X_test = None
        Y_train = np.concatenate([all_labels[i] for i in train_idxs])
        Y_test = np.concatenate([all_labels[i] for i in test_idxs])
        G_train = np.concatenate(
            [np.full(len(all_labels[i]), i) for i in train_idxs])
        G_test = np.concatenate(
            [np.full(len(all_labels[i]), i) for i in test_idxs])

        # FVs model
        fvs = FisherVectors()

        if os.path.isfile(train_fvs_path) and os.path.isfile(test_fvs_path):
            print('loading train fvs')
            X_train = joblib.load(train_fvs_path)
            print('loading test fvs')
            X_test = joblib.load(test_fvs_path)
            print('done')
        else:
            X_train, _ = fvs.train_fv_gmm_and_compute_fvs_from_files(
                train_files, bag_idx, feature_idx)
            X_test, _ = fvs.compute_fvs_from_files(
                test_files, bag_idx, feature_idx)
            joblib.dump(X_train, train_fvs_path)
            joblib.dump(X_test, test_fvs_path)

        if skip_train:
            continue

        assert len(Y_train) == len(X_train)
        assert len(G_train) == len(X_train)

        assert len(Y_test) == len(X_test)
        assert len(G_test) == len(X_test)

        if frame_scores_path:
            assert len(X_test) == len(test_map)

        # sample training set
        print('sampling training set')
        X_train, Y_train, G_train = sample_train_set(X_train, Y_train, G_train,
                                                     num_bags,
                                                     balance)
        fold_results['train_num_bags'] = len(X_train)
        fold_results['test_num_bags'] = len(X_test)

        # train
        print('starting training')
        train_results, train_scores = fisher_vectors.helpers.train(fvs, X_train, Y_train, G_train, svm_c)
        add_to_dict(fold_results, train_results, prefix='train')

        # test
        test_results, test_scores = fisher_vectors.helpers.test(fvs, X_test, Y_test)
        add_to_dict(fold_results, test_results, prefix='test')

        fold_results['C'] = fvs.svm.get_params()['C']
        results.append(fold_results)

        # save the scores
        for p in np.unique(G_test):
            scores[p] = test_scores[np.nonzero(G_test == p)]

        # output the scores
        if frame_scores_path:
            for i, (score, subject) in enumerate(zip(test_scores, G_test)):
                first_frame = int(all_info[subject][test_map[i], 2])
                last_frame = int(all_info[subject][test_map[i], 3])
                frame_scores[first_frame:last_frame + 1, subject] = score
            np.savetxt(frame_scores_path, frame_scores)

    return pd.DataFrame(results), scores