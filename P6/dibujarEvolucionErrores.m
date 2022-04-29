function dibujarEvolucionErrores(lambda,best_lambda,Etr,Ecv)
%Dibujar evolución tasas de error de entrenamiento y validación
    figure;
    grid on; hold on;
    ylabel('Tasa de error'); xlabel('Factor de regularización');

    plot(log10(lambda), Etr, 'r-', 'LineWidth',1);
    plot(log10(lambda), Ecv, 'b-', 'LineWidth',1);
    plot(log10(lambda(best_lambda)), Ecv(best_lambda), 'g.');
    legend('Tasa de error de entrenamiento', 'Tasa de error de validación', 'Punto ideal');
end

