(define (problem BLOCKS-4-0)
    (:domain BLOCKS)
    (:objects
        a b c d
    )
    (:init
        (clear c)
        (clear a)
        (clear b)
        (clear d)
        (ontable c)
        (ontable a)
        (ontable b)
        (ontable d)
        (handempty)
    )
    (:goal
        (and
            (on d c)
            (on c b)
            (on b a))
    )
)
