function [y_pred] = pred_sigmoid(X,theta)
    h = 1./(1+exp(-(X*theta)));
    h(:,1) = [];
    [~,y_pred] = max(h,[],2);
end

