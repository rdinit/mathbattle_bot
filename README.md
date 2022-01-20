# mathbattlebot

 Bot playing mathbattle game from Telegram messenger

* Installing:
    
    run in command line

    ```
    pip3 install -r requirements.txt
    ```
* Running:
    Example code
    ```
    Brain('model/v3',
                    (740, 180, 1420, 460),  # grab screen (x,y,x,y)
                    90,  # letter height (needed for correct resize)
                    (1070, 730),  # positin of restart button, this pixel should be green
                    (930, 740),  # left button coordinates
                    (1220, 740),  # right button coordinates
                    (750, 510, 650)  # progressbar (x, y, length)
                )
    brain.think(chck=True, loop_id=1)
    ```
    Config parameters personally for your screen
