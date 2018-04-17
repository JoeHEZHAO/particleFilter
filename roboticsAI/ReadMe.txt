Date: 04-10-2018

TODO: Enable Particle to be re-used for multiple experiments
TODO: Package prediction and correction for single line usage
TODO: Display results for multi-channel and multi-motion model


Update: Package prediction and correction ==> Done;
        Display results for multi-channel and multo-motion ==> Done
        Enable Particle tobe re-used for multiple experiments ===> Done

TODO: Add covariance matrix cross-channel
TODO: 


Date: 04-11-2018
#TODO: Add covariance matrix for cross-channel velocity and acceleration

#TODO: Using cross-channel acceleration information for propagation and prediction
#TODO: Compare with using single channel information only

Conclusion:
    1. Using cross-channel acceleration information gain stable results for sharp changing area;
    2. For not-quite-curving segments, cross-channel and original line are quite similar


Date: 04-16-2018

#1. writing backwards gradient computation as gradient_1_bk
#2. testing gradien_1_bk to see how does it goes

Conclusion:
    1. back-ward gradient is much less stable than center gradient
    2. When true data oscillate dramatically, the particle fitler prediction results is not good
    3. When true data is quite stationary, the particle filter result is good;
    4. So sum up, back-ward gradient is ok when data is stable; While advantage is it won't need any other information from future;

TODO: calculate backward acceleration and try;
TODO: re-design second order gradient method;

Attention: 
    1. np.gradient can only calculate first order gradient
    2. np.gradient(Y, 2), only means compute gradient with step size 2;
    3. second order gradient is to call above function twice np.grandient(np.gradient(Y, 2), 2)
    4. The actually second order gradient([0, 10]) is actually muuch more un stable than gradient_1([0, 1])
    5. Re-try second order gradient to testify 

Conclustion:
    1. Correct second order gradient computation
    2. The result does not have too much changing
    3. Using backwards also does not bring in too many trouble
    4. So, it is fair to say that backwards gradient is ok

Attention:
    1. Actually, using gradient_bk for pure prediction brings better performance;
    2. So, gradient_bk is totally better
    3. From now on, use gradient_1_bk 

