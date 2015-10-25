function responsibility = independent_sampling( data, N, M, mu )
    
    data = [(1 : N)' data];
    data_pos_indices = data(data(:,end) == 1, 1);
    data_neg_indices = data(data(:,end) == -1, 1);
    N_pos = round((size(data_pos_indices,1) / N) * mu * N);
    N_neg = round((size(data_neg_indices,1) / N) * mu * N);
    pos_min_index = min(data_pos_indices);
    pos_max_index = max(data_pos_indices);
    neg_min_index = min(data_neg_indices);
    neg_max_index = max(data_neg_indices);
    
    responsibility = 0 * zeros(N,M);
    for j = 1 : M
        for i = 1 : N_pos
            while 1 == 1
                rand_i = randi(pos_max_index - pos_min_index + 1) + pos_min_index - 1;
                if responsibility(rand_i,j) == 0
                    responsibility(rand_i,j) = 1;
                    break;
                end
            end
        end
        
        for i = 1 : N_neg
            while 1 == 1
                rand_i = randi(neg_max_index - neg_min_index + 1) + neg_min_index - 1;
                if responsibility(rand_i,j) == 0
                    responsibility(rand_i,j) = 1;
                    break;
                end
            end
        end
    end

    responsibility = (responsibility == 1);
    
    for i = 1 : N
        if sum(responsibility(i,:)) == 0
            fprintf('Bagging failed!\n');
            responsibility = -1;
            break;
        end
    end

end

