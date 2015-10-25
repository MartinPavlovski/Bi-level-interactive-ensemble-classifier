function [ ] = call_interactive_ensemble( DATASETNAME, M, M_subset, mu, boosting_algorithm )

train_set_path = strcat('Datasets/',DATASETNAME,'_train.csv');
test_set_path = strcat('Datasets/',DATASETNAME,'_test.csv');

% Load train data
data = load(train_set_path);
N = size(data,1);
for i = 1 : N
    if data(i,end) <= 0
        data(i,end) = -1;
    else
        data(i,end) = 1;
    end
end
X = data(:,1:end-1);
y = data(:,end);

% Load test data
data_test = load(test_set_path);
N_test = size(data_test,1);
for i = 1 : N_test
    if data_test(i,end) <= 0
        data_test(i,end) = -1;
    else
        data_test(i,end) = 1;
    end
end
X_test = data_test(:,1:end-1);
y_test = data_test(:,end);


responsibility = zeros(N,M);
while 1 == 1
    responsibility = independent_sampling(data,N,M,mu);
    if responsibility ~= -1
        break;
    end
end
fprintf('BAGGING DONE!\n');


% Train
models = cell(M,1);
for j = 1 : M

    misses = zeros(N,j);
    for k = 1 : j
        X_subset = X(responsibility(:,k),:);
        y_subset = y(responsibility(:,k),:);
        models{k} = fitensemble(X_subset,y_subset,boosting_algorithm,M_subset,'tree');
        predictions = predict(models{k},X);
        misses(:,k) = double(y ~= predictions);
    end



    Y_M = zeros(N,1);
    for k = 1 : j
        Y_M = Y_M + predict(models{k},X);
    end
    Y_M = sign(Y_M);
    Y_M(Y_M == 0) = 1;
    train_error = 100 * sum(y ~= Y_M) / N;



    % Test
    Y_M = zeros(N_test,1);
    for k = 1 : j
        Y_M = Y_M + predict(models{k},X_test);
    end
    Y_M = sign(Y_M);
    Y_M(Y_M == 0) = 1;
    test_error = 100 * sum(y_test ~= Y_M) / N_test;



    % Print errors
    fprintf('%d,%f,%f\n',j,train_error,test_error);


    % Rescheduling instances in different subsets
    if j > 1
        for i = 1 : N
            if misses(i,j) == 1
                for k = 1 : (j - 1)
                    jj = (j - k);
                    if responsibility(i,jj) == 0
                        model_from_before = models{j};
                        predictions = predict(model_from_before,X);
                        subset_from_error_before = sum(y ~= predictions) / N;

                        model_to_before = models{jj};
                        predictions = predict(model_to_before,X);
                        subset_to_error_before = sum(y ~= predictions) / N;

                        responsibility(i,j) = 0;
                        responsibility(i,jj) = 1;

                        X_subset = X(responsibility(:,j),:);
                        y_subset = y(responsibility(:,j),:);
                        model_from_after = fitensemble(X_subset,y_subset,boosting_algorithm,M_subset,'tree');
                        predictions = predict(model_from_after,X);
                        subset_from_error_after = sum(y ~= predictions) / N;

                        X_subset = X(responsibility(:,jj),:);
                        y_subset = y(responsibility(:,jj),:);
                        model_to_after = fitensemble(X_subset,y_subset,boosting_algorithm,M_subset,'tree');
                        predictions = predict(model_to_after,X);
                        subset_to_error_after = sum(y ~= predictions) / N;

                        if (subset_from_error_after < subset_from_error_before && subset_to_error_after < subset_to_error_before) || ...
                           (subset_from_error_after == subset_from_error_before && subset_to_error_after < subset_to_error_before) || ...
                           (subset_from_error_after < subset_from_error_before && subset_to_error_after == subset_to_error_before)
                            models{j} = model_from_after;
                            models{jj} = model_to_after;
                            break;
                        else
                            responsibility(i,j) = 1;
                            responsibility(i,jj) = 0;
                        end
                    end
                end
            end
        end
    end

end

fprintf('\n\n\n === END === \n\n\n');



end

